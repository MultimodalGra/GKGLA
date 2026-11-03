import math
import torch
from torch import nn
from torch.nn import functional as F


class ILRPResidualBranch(nn.Module):

    def __init__(self, in_features, out_features, rank=8):
        super(ILRPResidualBranch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, input_x):

        out = self.lora_B(self.lora_A(input_x))
        return out


class ILRP:
    def _init_base_projection_from_clip(self):

        with torch.no_grad():
            nn.init.eye_(self.base_proj_img.weight)
            nn.init.eye_(self.base_proj_text.weight)

        self.base_proj_img.weight.requires_grad = False
        self.base_proj_text.weight.requires_grad = False

    def _sum_lora(self, x, branches):

        projection_mode = getattr(self, "ablation_projection_mode", "full")
        if projection_mode == "none":
            branches = branches[:0]
        elif projection_mode == "old":
            branches = branches[:-1]
        elif projection_mode != "full":
            raise ValueError(
                "Unknown ablation_projection_mode: {}".format(projection_mode)
            )
        if len(branches) == 0:
            return 0.0
        out = 0.0
        for b in branches:
            out = out + self.lora_scale * b(x)
        return out

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        base_out = self.base_proj_img(basic_img_features)
        image_feas = base_out + self._sum_lora(basic_img_features, self.projs_img)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        base_out = self.base_proj_text(basic_text_features)
        text_feas = base_out + self._sum_lora(basic_text_features, self.projs_text)
        return F.normalize(text_feas, dim=-1) if normalize else text_feas

    def encode_prototypes(self, normalize: bool = False):
        self.img_prototypes = self.img_prototypes.to(self._device)
        base_out = self.base_proj_img(self.img_prototypes)
        image_feas = base_out + self._sum_lora(self.img_prototypes, self.projs_img)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def extend_item(self):
        return ILRPResidualBranch(
            self.feature_dim, self.feature_dim, rank=self.lora_rank
        ).to(self._device)

    def extend_task(self):

        self._task_count += 1

        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())
