
import importlib.metadata as metadata
import json, platform, sys, torch
from pathlib import Path
import transformers, tokenizers, peft, trl
from common.data import load_yaml, encode_prompt_response
from common.models import reference_mode, make_lora_config
from common.generation import response_sequence_logprobs, response_token_logprobs
from task1_dpo.dpo import dpo_loss
import task1_dpo.train, task1_dpo.verify_checkpoint
from task1_dpo.support import validate_config
from task1_dpo.evaluation_support import evaluation_plan
expected = {"transformers": "4.57.1", "tokenizers": "0.22.1",
            "peft": "0.17.1", "trl": "0.27.2", "bitsandbytes": "0.50.2"}
versions = {name: metadata.version(name) for name in expected}
assert versions == expected, versions
assert str(torch.__version__) == "2.11.0+cpu", torch.__version__
assert not torch.cuda.is_available(), "Unexpected GPU allocation."
cfg = load_yaml("configs/dpo.yaml")
validate_config(cfg)
scopes = [evaluation_plan(cfg, name)["name"] for name in
          ("standard", "beta_003", "beta_010", "beta_030", "length_balanced", "sft")]
record = {"status": "CPU_PACKAGES_OK", "python": platform.python_version(),
    "torch": str(torch.__version__), "packages": versions,
    "course_and_task1_imports": "PASS", "approved_config": "PASS",
    "evaluation_scopes": scopes, "gpu_available": False,
    "model_weights_loaded": False, "official_training_started": False}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
