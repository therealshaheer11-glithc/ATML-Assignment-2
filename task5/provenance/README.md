# Source and evidence provenance

The course source is from commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`
of https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining . The copied common
loading/generation helpers, exact verifier, pairwise judge, configs and course
requirements remain byte-identical inside the frozen release. Its manifest SHA
is `d6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87`.
Task 5 orchestration, inspections, analysis and package tooling were developed
with Codex assistance. Qwen models, adapters and dataset identities are pinned in
the original bindings. Task 4 wheel inspection utilities are attributed there.

`NOTEBOOK_AUDIT.json` binds the supplied notebook by SHA-256, records each cell's
source/output hash and recovered errors, and distinguishes AST-equivalent cells
from the reviewed nonsemantic edits. `notebook_sources/` is historical evidence;
run the modular release or documented package commands instead of executing all
historical cells. `logs/notebook_execution_selected.log` contains selected status,
count, test and hash lines from the notebook outputs; it is not a full stdout dump.

The private CPU evidence bundle SHA is
`8bace113620a0080808a50df9b9ca296e881cb7ac430c118b6c73f339c030067`.
Its original manifest is retained as an external evidence index. Entries naming
private files do not mean those files are included in this GitHub payload.
Original copied reports/results keep their exact bytes. Public metric records
are explicit projections; their schema and optional private verification are in
`verify_package.py`. Private source questions, diagnostic texts and full generated
responses have not been relabeled as public records.

No raw response text was changed, no judgment was relabeled, and no metric was
rescored with a new parser. Qualitative examples were selected after explicit
user approval using original source order. Authored interpretations in RESULTS.md
are analysis of this recorded run, not independently collected human labels.
