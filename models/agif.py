import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class HHRM:
    @staticmethod
    def retrieve(history, norm, pseudo_query, mode="learned"):
        values = torch.cat(history, dim=1)
        keys = norm(values)
        query = pseudo_query.unsqueeze(0).expand(values.shape[0], -1, -1)
        weights = F.softmax(torch.bmm(query, keys.transpose(1, 2)), dim=-1)
        if mode == "uniform":
            weights = torch.full_like(weights, 1.0 / values.shape[1])
        elif mode in {"last_only", "initial_only"}:
            weights = torch.zeros_like(weights)
            weights[..., -1 if mode == "last_only" else 0] = 1.0
        elif mode != "learned":
            raise ValueError(f"Unknown ablation_history_mode: {mode}")
        return torch.bmm(weights, values), weights


class AGIF:
    def _run_agif(
        self,
        image_features,
        text_features,
        prototype_features,
        context_prompts,
        return_diagnostics=False,
    ):
        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_all_prompts = context_prompts.shape[0]

        image_features = image_features.view(
            image_features.shape[0], 1, self.feature_dim
        )
        text_features = text_features.view(1, len_texts, self.feature_dim)
        prototype_features = prototype_features.view(1, len_protos, self.feature_dim)

        text_features = text_features.expand(image_features.shape[0], -1, -1)
        prototype_features = prototype_features.expand(image_features.shape[0], -1, -1)

        if len_all_prompts > 0:
            context_prompts = context_prompts.view(1, len_all_prompts, self.feature_dim)
            context_prompts = context_prompts.expand(image_features.shape[0], -1, -1)
            feat = torch.cat(
                [image_features, text_features, prototype_features, context_prompts],
                dim=1,
            )
        else:
            feat = torch.cat([image_features, text_features, prototype_features], dim=1)

        img_history = [image_features]
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "image_layers": [image_features[:, 0, :]],
                "attn_weights": [],
                "gate_vectors": [],
            }

        active_layers = int(getattr(self, "ablation_num_layers", self.num_layers))
        if active_layers < 0 or active_layers > self.num_layers:
            raise ValueError(
                "ablation_num_layers must be in [0, {}]".format(self.num_layers)
            )
        history_mode = getattr(self, "ablation_history_mode", "learned")
        gate_mode = getattr(self, "ablation_gate_mode", "learned")

        for i in range(active_layers):
            sel_attn = self.sel_attn_layers[i]

            gamma = self.gamma_layers[i]
            if self.debug_print:
                print(f"==== Task {self._task_count} Layer {i} raw gamma vector ====")
                print(gamma.detach().cpu())
                print(f"Layer {i} gamma mean value: {gamma.mean().item():.6f}")

            img_base, attn_weights = HHRM.retrieve(
                img_history,
                self.attn_res_norms[i],
                self.pseudo_queries[i],
                history_mode,
            )

            sa_out = sel_attn(feat, feat, feat)

            img_sa_out = sa_out[:, 0:1, :]
            other_sa_out = sa_out[:, 1:, :]
            other_feat_base = feat[:, 1:, :]

            gate_vec = torch.sigmoid(gamma)
            if self.debug_print:
                print(f"==== Layer {i} sigmoid gate vector ====")
                print(gate_vec.detach().cpu())
                print(f"Layer {i} gate_vec mean: {gate_vec.mean().item():.6f}\n")

            if gate_mode == "initial":
                initial_gate = float(torch.sigmoid(gamma.new_tensor(-2.0)))
                gate_vec = torch.full_like(gamma, initial_gate)
            elif gate_mode != "learned":
                raise ValueError("Unknown ablation_gate_mode: {}".format(gate_mode))

            new_img_feat = (1 - gate_vec) * img_base + gate_vec * img_sa_out

            new_other_feat = (1 - gate_vec) * other_feat_base + gate_vec * other_sa_out

            feat = torch.cat([new_img_feat, new_other_feat], dim=1)

            img_history.append(new_img_feat)

            if return_diagnostics:
                diagnostics["image_layers"].append(new_img_feat[:, 0, :])
                diagnostics["attn_weights"].append(attn_weights[:, 0, :])
                diagnostics["gate_vectors"].append(gate_vec)

        kv_final = feat[:, : feat.shape[1] - len_all_prompts, :]
        img_out = kv_final[:, 0]
        txt_out = kv_final[:, 1 : 1 + len_texts].mean(0)
        proto_out = kv_final[:, 1 + len_texts : 1 + len_texts + len_protos].mean(0)

        if return_diagnostics:
            return img_out, txt_out, proto_out, diagnostics
        return img_out, txt_out, proto_out
