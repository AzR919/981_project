"""The weights sent to vLLM must carry the current LoRA update (needs torch, transformers, peft; CPU only)."""

import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import copy
import os
import sys
import unittest

import torch

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

try:
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # the laptop has no peft; run in the lab environment
    LoraConfig = None

from llm.train import vllm_weights  # noqa: E402


def tiny_lora():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2)
    base = LlamaForCausalLM(config).to(torch.bfloat16)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if "lora_B" in name:
                param.data.normal_(0, 0.5)  # as if trained: a nonzero update
    return model, before


@unittest.skipIf(LoraConfig is None, "peft not installed")
class TestLoraSync(unittest.TestCase):
    def test_update_reaches_vllm_weights(self):
        model, before = tiny_lora()
        sent = dict(vllm_weights(model, lora=4))
        name = "model.layers.0.self_attn.q_proj.weight"
        self.assertFalse(torch.equal(sent[name], before[name]))
        self.assertTrue(torch.equal(sent["model.layers.0.self_attn.k_proj.weight"],
                                    before["model.layers.0.self_attn.k_proj.weight"]))

    def test_matches_merge_and_unload(self):
        model, _ = tiny_lora()
        sent = dict(vllm_weights(model, lora=4))
        merged = copy.deepcopy(model).merge_and_unload().state_dict()
        self.assertEqual(set(sent), set(merged))
        for k in sent:
            torch.testing.assert_close(sent[k].float(), merged[k].float(), rtol=2e-2, atol=1e-2)

    def test_base_weights_untouched_and_copies_returned(self):
        model, before = tiny_lora()
        sent = dict(vllm_weights(model, lora=4))
        base = model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight
        self.assertTrue(torch.equal(base, before["model.layers.0.self_attn.q_proj.weight"]))
        snapshot = sent["model.layers.0.self_attn.k_proj.weight"].clone()
        model.base_model.model.model.layers[0].self_attn.k_proj.weight.data.add_(1.0)
        self.assertTrue(torch.equal(sent["model.layers.0.self_attn.k_proj.weight"], snapshot))

    def test_full_finetuning_returns_copies(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=1, num_attention_heads=4,
                                             num_key_value_heads=2)).to(torch.bfloat16)
        sent = dict(vllm_weights(model, lora=0))
        snapshot = sent["lm_head.weight"].clone()
        model.lm_head.weight.data.add_(1.0)
        self.assertTrue(torch.equal(sent["lm_head.weight"], snapshot))


if __name__ == "__main__":
    unittest.main()
