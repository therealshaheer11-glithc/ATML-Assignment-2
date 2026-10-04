
import json, sys
from pathlib import Path
from common.data import load_yaml
from task1_dpo.support import file_sha, save_json, validate_config
from task1_dpo.reward_loading import compatible_reward_config
stage = Path(sys.argv[1])
cfg = load_yaml("configs/dpo.yaml")
validate_config(cfg)
preparation = json.loads(Path("docs/task1_model_preparation.json").read_text())
models = Path(preparation["drive_models_dir"])
manifest_path = models / "model_manifest.json"
assert file_sha(manifest_path) == preparation["model_manifest_sha256"]
manifest = json.loads(manifest_path.read_text())
entry = manifest["reward"]
assert (entry["model_id"], entry["revision"]) == (cfg["reward_model"], cfg["reward_model_revision"])
folder = (models / entry["directory"]).resolve()
assert models.resolve() in folder.parents
path = folder / "config.json"
assert file_sha(path) == entry["sha256"]["config.json"] == "d04c63bcaf27f49f7fea8c3c1287ba4078056f3b3ed35d2635c6caef7dbad9fb"
effective, report = compatible_reward_config(path)
assert report["changed_fields"] == {"rope_theta": {"before": 10000.0, "after": 1000000.0}}
assert effective.max_position_embeddings >= cfg["reward_max_length"]
assert effective.num_labels == 1
report.update({"reward_model": cfg["reward_model"], "reward_revision": cfg["reward_model_revision"],
    "model_manifest_sha256": file_sha(manifest_path), "weights_loaded": False,
    "official_training_started": False, "gpu_available": False})
save_json(stage / "real_configuration_check.json", report)
print("PASS: pinned reward setting corrected in memory: 10000 -> 1000000.")
print("PASS: only rope_theta changed; original config hash unchanged; declared frequencies match.")
print("PASS: approved input limits/configuration unchanged; no model weights loaded.")
