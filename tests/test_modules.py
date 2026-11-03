import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.agif import HHRM, RMSNorm
from models.network import GKGLANet


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 8
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, x):
        return x

    def encode_text(self, x):
        return x


def make_network(prompts=3):
    args = {
        "device": [torch.device("cpu")],
        "lora_rank": 2,
        "lora_alpha": 4,
        "fusion_num_layers": 3,
        "context_prompt_length_per_task": prompts,
    }
    with patch(
        "models.backbone.build_backbone", return_value=(SyntheticBackbone(), None, None)
    ):
        return GKGLANet(args, False)


class ModuleTests(unittest.TestCase):
    def test_identity_and_freezing(self):
        net = make_network()
        net.extend_task()
        net.update_context_prompt()
        x = torch.randn(4, 8)
        torch.testing.assert_close(net.encode_image(x), x, rtol=0, atol=0)
        torch.testing.assert_close(net.encode_text(x), x, rtol=0, atol=0)
        net.extend_task()
        net.update_context_prompt()
        net.freeze_projection_weight_new()
        self.assertTrue(all(not p.requires_grad for p in net.projs_img[0].parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in net.projs_text[0].parameters())
        )
        self.assertTrue(all(p.requires_grad for p in net.projs_img[-1].parameters()))
        self.assertFalse(net.base_proj_img.weight.requires_grad)
        self.assertFalse(net.context_prompts[0].requires_grad)
        self.assertTrue(net.context_prompts[-1].requires_grad)

    def test_historical_retrieval(self):
        history = [torch.randn(2, 1, 8) for _ in range(3)]
        query = torch.zeros(1, 8)
        base, weights = HHRM.retrieve(history, RMSNorm(8), query)
        torch.testing.assert_close(base, torch.cat(history, 1).mean(1, keepdim=True))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 1))
        for mode, index in [("last_only", -1), ("initial_only", 0)]:
            base, _ = HHRM.retrieve(history, RMSNorm(8), query, mode)
            torch.testing.assert_close(base, history[index])

    def test_fusion_gradient_and_optional_prompts(self):
        for prompts in (0, 3):
            net = make_network(prompts)
            net.extend_task()
            net.update_context_prompt()
            net.update_prototype(5)
            net.img_prototypes = torch.randn(5, 8)
            output = net(torch.randn(2, 8), torch.randn(5, 8))
            self.assertEqual(output[0].shape, (2, 8))
            self.assertEqual(output[1].shape, (5, 8))
            loss = sum(t.square().sum() for t in (output[0], output[1], output[3]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(net.projs_img[-1].lora_B.weight.grad)
            self.assertIsNotNone(net.gamma_layers[-1].grad)
            self.assertIsNotNone(net.pseudo_queries[-1].grad)


if __name__ == "__main__":
    unittest.main()
