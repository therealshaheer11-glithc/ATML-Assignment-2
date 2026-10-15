"""CPU checks of the approved translation and pinned-source safeguards.

Config files are fixtures only. No pretrained model weights are loaded.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from task1_dpo import reward_loading as reward

RAW = {"model_type": "qwen2", "architectures": ["Qwen2ForSequenceClassification"],
       "hidden_size": 16, "num_attention_heads": 2, "num_key_value_heads": 2,
       "num_hidden_layers": 1, "intermediate_size": 32, "vocab_size": 32,
       "max_position_embeddings": 32768, "id2label": {"0": "LABEL_0"},
       "label2id": {"LABEL_0": 0},
       "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}}

class RewardLoadingTests(unittest.TestCase):
    def test_translation_changes_only_theta_and_actual_rotary_constants(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(RAW))
            before = path.read_bytes()
            config, record = reward.compatible_reward_config(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(record["changed_fields"], {"rope_theta": {"before": 10000.0, "after": 1000000.0}})
            dim = RAW["hidden_size"] // RAW["num_attention_heads"]
            expected = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2).float() / dim))
            rotary = Qwen2RotaryEmbedding(config, device=torch.device("cpu"))
            torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)

    def test_conflicting_unknown_and_nonfinite_settings_are_rejected(self):
        variants = []
        for change in ({"rope_theta": 10000.0}, {"rope_type": "linear"},
                       {"extra": True}, {"rope_theta": float("inf")}):
            raw = copy.deepcopy(RAW)
            if change == {"rope_theta": 10000.0}:
                raw.update(change)
            else:
                raw["rope_parameters"].update(change)
            variants.append(raw)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for raw in variants:
                with self.subTest(raw=raw):
                    path.write_text(json.dumps(raw))
                    with self.assertRaises(ValueError):
                        reward.compatible_reward_config(path)

    def test_cpu_guard_stops_before_snapshot_or_weights_loading(self):
        with patch.object(reward.torch.cuda, "is_available", return_value=False), \
             patch.object(reward, "reward_source") as source, \
             patch.object(reward.AutoModelForSequenceClassification, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                reward.load_pinned_reward({}, "/does-not-exist")
            source.assert_not_called()
            load.assert_not_called()

    def test_pinned_source_rejects_file_drift_and_manifest_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "reward"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(RAW))
            (folder / "model.safetensors").write_bytes(b"synthetic-weight-file-not-loaded")
            manifest = {"reward": {"model_id": "fixture", "revision": "pinned",
                "directory": "reward", "sha256": {p.name: reward.file_sha(p) for p in folder.iterdir()}}}
            manifest_path = root / "model_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            preparation = root / "preparation.json"
            preparation.write_text(json.dumps({"model_manifest_sha256": reward.file_sha(manifest_path)}))
            cfg = {"reward_model": "fixture", "reward_model_revision": "pinned"}
            with patch.object(reward, "validate_config"), patch.object(reward, "repo_path", return_value=preparation):
                self.assertEqual(reward.reward_source(cfg, root), folder.resolve())
                (folder / "model.safetensors").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "file changed"):
                    reward.reward_source(cfg, root)
                manifest_path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    reward.reward_source(cfg, root)

if __name__ == "__main__":
    unittest.main()
