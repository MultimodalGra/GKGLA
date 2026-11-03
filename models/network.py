import copy
import torch
from torch import nn
from convs.attention import MultiHeadAttention
from models.ilrp import ILRP
from models.agif import AGIF, RMSNorm
from utils.toolkit import get_attribute


class GKGLANet(ILRP, AGIF, nn.Module):
    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def __init__(self, args, pretrained):
        nn.Module.__init__(self)
        from models.backbone import build_backbone

        self.convnet, self.preprocess, self.tokenizer = build_backbone(args)
        self.fc = None
        self.args = args
        self._device = args["device"][0]
        self.context_prompt_length_per_task = get_attribute(
            self.args, "context_prompt_length_per_task", 3
        )

        self.use_context_prompt = self.context_prompt_length_per_task > 0
        self.debug_print = bool(get_attribute(self.args, "debug_print", False))

        self._task_layer_stats = []

        self.lora_rank = int(get_attribute(self.args, "lora_rank", 64))
        self.lora_alpha = float(get_attribute(self.args, "lora_alpha", 128.0))
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        self.lora_scale = self.lora_alpha / self.lora_rank

        self._task_count = 0

        self.base_proj_img = nn.Linear(
            self.feature_dim, self.feature_dim, bias=False
        ).to(self._device)
        self.base_proj_text = nn.Linear(
            self.feature_dim, self.feature_dim, bias=False
        ).to(self._device)
        self.use_clip_proj_init = bool(
            get_attribute(self.args, "use_clip_proj_init", False)
        )
        self._init_base_projection_from_clip()

        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()

        self.num_layers = int(get_attribute(self.args, "fusion_num_layers", 5))
        if self.num_layers < 1:
            raise ValueError("fusion_num_layers must be at least 1")
        self.sel_attn_layers = nn.ModuleList()

        self.fuse_norm_layers = nn.ModuleList()

        self.gamma_layers = nn.ParameterList()

        for _ in range(self.num_layers):
            self.sel_attn_layers.append(
                MultiHeadAttention(
                    1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1
                )
            )

            self.fuse_norm_layers.append(nn.LayerNorm(self.feature_dim))

            self.gamma_layers.append(nn.Parameter(torch.tensor(1.0)))

        self.pseudo_queries = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.feature_dim))
                for _ in range(self.num_layers)
            ]
        )

        self.attn_res_norms = nn.ModuleList(
            [RMSNorm(self.feature_dim) for _ in range(self.num_layers + 1)]
        )

        self.gamma_layers = nn.ParameterList(
            [
                nn.Parameter(torch.ones(self.feature_dim) * -2.0)
                for _ in range(self.num_layers)
            ]
        )

        self.img_prototypes = None
        self.context_prompts = nn.ParameterList()

    def update_prototype(self, nb_classes):
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat(
                [
                    copy.deepcopy(self.img_prototypes).to(self._device),
                    torch.zeros(nb_classes - nb_output, self.feature_dim).to(
                        self._device
                    ),
                ]
            ).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(
                self._device
            )
        print(
            "update prototype, now we have {} prototypes".format(
                self.img_prototypes.shape[0]
            )
        )

    def update_context_prompt(self):
        if not self.use_context_prompt:
            return
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(
            nn.Parameter(
                torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(
                    self._device
                )
            )
        )
        print(
            "update context prompt, now we have {} context prompts".format(
                len(self.context_prompts) * self.context_prompt_length_per_task
            )
        )
        self.context_prompts.to(self._device)

    def get_context_prompts(self):
        if len(self.context_prompts) == 0:
            return torch.zeros(0, self.feature_dim, device=self._device)
        return torch.cat([item for item in self.context_prompts], dim=0)

    def freeze_projection_weight_new(self):

        if len(self.projs_img) > 1:
            for i in range(len(self.projs_img) - 1):
                for p in self.projs_img[i].parameters():
                    p.requires_grad = False
                for p in self.projs_text[i].parameters():
                    p.requires_grad = False

        if len(self.projs_img) > 0:
            for p in self.projs_img[-1].parameters():
                p.requires_grad = True
            for p in self.projs_text[-1].parameters():
                p.requires_grad = True

        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = i == len(self.context_prompts) - 1

        trainable_lists = [self.gamma_layers, self.pseudo_queries, self.attn_res_norms]
        for module_list in trainable_lists:
            for item in module_list:
                if isinstance(item, nn.Parameter):
                    item.requires_grad = True
                elif isinstance(item, nn.Module):
                    for p in item.parameters():
                        p.requires_grad = True

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        prototype_features = self.encode_prototypes(normalize=True)
        context_prompts = self.get_context_prompts()

        image_features, text_features, prototype_features = self._run_agif(
            image_features, text_features, prototype_features, context_prompts
        )

        return (
            image_features,
            text_features,
            self.convnet.logit_scale.exp(),
            prototype_features,
        )

    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototypes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            image_features, text_features, prototype_features = self._run_agif(
                image_features, text_features, prototype_features, context_prompts
            )
            if self.debug_print:
                print("\n======== All layers pseudo_queries summary ========")
                for layer_idx, query_param in enumerate(self.pseudo_queries):
                    print(f"Layer {layer_idx} pseudo_query:")
                    print(query_param.detach().cpu())
                    print(f"L2 Norm: {torch.norm(query_param).item():.6f}\n")
            return (
                image_features,
                text_features,
                self.convnet.logit_scale.exp(),
                prototype_features,
            )
        else:
            return (
                image_features,
                text_features,
                self.convnet.logit_scale.exp(),
                prototype_features,
            )

    def forward_transformer_with_diagnostics(self, image_features, text_features):

        prototype_features = self.encode_prototypes(normalize=True)
        context_prompts = self.get_context_prompts()
        image_features, text_features, prototype_features, diagnostics = self._run_agif(
            image_features,
            text_features,
            prototype_features,
            context_prompts,
            return_diagnostics=True,
        )
        return (
            image_features,
            text_features,
            self.convnet.logit_scale.exp(),
            prototype_features,
            diagnostics,
        )
