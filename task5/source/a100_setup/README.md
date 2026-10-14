# Task 5 approved isolated A100 setup

Run `prepare_a100.py` from this sealed source directory in a Colab A100 runtime after mounting the existing Drive project. The single Colab wrapper verifies the uploaded source before execution. This setup is authorized by the user's earlier "okay go ahead" environment-plan approval; CPU readiness is established by the subsequent successful combined preflight report.

The setup preserves the exact 84 package versions already approved. It verifies the successful CPU report and readiness contract, freshly inspects the A100/platform/wheel cache, requires the approved disk headroom, creates or verifies `/content/task5_isolated_runtime_v1`, bootstraps cached pip 24.1.2 and performs a strictly offline, hashed installation without dependency resolution. It checks complete isolation, the exact package inventory and `pip check` in fresh subprocesses.

The GPU probe imports the required packages, runs two tiny float16 CUDA/NF4 checks, reads the pinned local tokenizer files and verifies 400 full policy prompts plus 80 diagnostic judge prompts against the model context budgets. It also runs the existing 44 synthetic evaluation tests in the isolated runtime. It does not instantiate any pretrained model, load model weight tensors, generate policy answers, judge responses, train, publish, repair the Colab base or download new packages/models.

Save evidence to Drive and stop on failure. Existing unowned environments are preserved. No versions, inference settings or frozen data are changed as a fallback. The known notebook-only plotting backend override is removed only from child environments, as approved; driver settings and the parent notebook environment are retained.

`guards.py`, `inspect_python.py` and `test_guards.py` are unchanged utilities reused from the earlier Task 4 preparation, with their historical hashes retained in this source manifest. Historical Task 4 approval is not used as Task 5 authorization. `APPROVED_PROPOSAL.json` is the unchanged original proposal, whose original PENDING label is historical; `APPROVAL.json` records the explicit subsequent user approval.

Successful setup saves `ISOLATED_ENVIRONMENT_CONTRACT.json` for the sealed Task 5 evaluation release. Pretrained model loading, the 1,200 policy answers, and the 880 AI comparison events remain separately approval gated. Keep the successful A100 runtime connected and return the report before the execution step.
