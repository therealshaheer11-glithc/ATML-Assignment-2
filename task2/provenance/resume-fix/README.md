# Task 2 JSON metadata resume repair

The original resume comparison used the live Transformers reward configuration, whose `id2label` dictionary has integer key `0`. Saving a contract as JSON changes that key to string `"0"`. Loading and comparing the saved contract to the live Python dictionary therefore falsely reported a changed evaluation contract. This was reproduced on CPU with the pinned reward configuration SHA256 `d04c63bcaf27f49f7fea8c3c1287ba4078056f3b3ed35d2635c6caef7dbad9fb`.

`Task2_GPU_Resume.ipynb` embeds the small compatibility module and installs it as an additional script. It uses the original GPU launcher with an in-memory wrapper around the evaluation call. The live contract is converted to precisely its stored JSON representation before the original strict comparison runs. No field is removed or ignored. Genuine changes to settings, source identities, adapters, or reward configuration still stop and print the differing fields.

The original Chunk 1–3 source files, installation hash manifests, job contract, completed training endpoints and archived evaluation records are preserved. The patch runs only after verifying the exact sealed evaluation-module hash. Original runtime validation checks all other sealed sources as before. It does not alter model loading, loss, optimizer, seeds, generation, update/evaluation budgets, or selection. The original session orchestrator retains the approved 120-minute scheduling limit, completed-unit rescue, verified session backup and Drive flush/disconnection gate.

The fix source checksum and explanation are saved to `experiment_v1/compatibility_fixes/json_contract_v1.json`, and activation is recorded in the saved GPU-session log. Existing results can resume without regeneration; an in-flight prompt not uploaded before disconnection may repeat from the last verified boundary. This is a mechanical persistence correction, not a new discretionary experimental choice.

Run the replacement notebook on A100; no ZIP upload or repeated CPU preparation is required. Do not rerun the uncorrected launcher. If the replacement reports real field differences, retain the logs and investigate them rather than deleting results or relaxing the guard.

The CPU regression suite specifically covers resuming an archive written by the original code, unchanged archive bytes, no repeated generation, retained rejection of actual settings changes, the original-source guard, and subprocess routing. The local real-GPU resume is not executed here.
