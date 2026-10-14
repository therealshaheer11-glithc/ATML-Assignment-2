# Repository layout — Task 5 integration

Task 5 now lives under `task5/`, with read-only CPU verification at its top level,
immutable executed Python releases under `source/`, metric/label records under
`records/`, fixed IDs under `splits/`, and checked results under `results/`.
The notebook's recoveries and source hashes are documented in `provenance/`.

Previous navigation, the combined verifier, this document and the maintenance
manifest are retained byte-for-byte in `archive/repository-maintenance-20261014-task5/`.
The root `task5_feedback/` is retained in place as original course reference,
including scaffolds; use the complete isolated Task 5 source instead. No old
files are deleted or moved. Tasks 1–4 and all original scientific manifests stay
unchanged. Their historical status text describes their earlier publication.

The combined verifier now invokes Task 5's CPU verifier and includes its declared
files in the existing optional stage/index checks. Check, apply, stage, commit
and push remain separate user-authorized actions. The package installer checks
the live remote and exact clean baseline `d1def5ccd50d609e797ebcac0759564b3a3bb8e5`,
verifies every source/destination hash before applying, and rolls back file bytes
on a caught failure. Process termination or power loss can still leave a partial
application requiring review. It never resets, stashes, deletes tracked history,
stages, commits or pushes.

Full raw course data, model weights, private review bundles and outer delivery
ZIPs remain outside Git. Task 5 public result reproduction operates on saved
score/length records; optional private-bundle verification recomputes exact scores
from full text. The combined eight-page report and LMS submission remain separate.
