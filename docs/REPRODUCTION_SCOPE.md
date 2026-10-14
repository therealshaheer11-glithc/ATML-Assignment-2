# What can be reproduced from this repository?

## Available now

`python3 -B scripts/reproduce_saved_results.py --output outputs/reproduced`
checks all publication files and rebuilds each task's saved-result analysis on CPU.
It requires only Python's standard library and an unused output directory. Each
subprocess has a saved log and the final receipt records hashes and exit codes.
Task 5 public reproduction uses saved score records; the original private bundle
is needed to recompute scores from full answers and gold finals.

## Original model execution

The root README now lists the implemented experiment entry points for each task.
The original source, configs, split IDs, model revisions, frozen manifests and
historical launchers are retained. Task 1 and Task 3 have public asset preparation
paths. Later safety evaluation requires the standard adapters trained in Tasks
1–3, not public base models alone.

## Remaining portability work

Tasks 2, 4 and 5 contain guards tied to original preparation receipts, Drive paths,
environment contracts or prior checkpoints. Listing their commands does not make
those prerequisites available to a new reader. A fully portable fresh-experiment
bootstrap is **not yet implemented or validated** for those workflows.

A proper follow-up must create new receipts by actually downloading/verifying
the pinned public inputs and checking the new environment; it must distinguish
a new reproduction from the frozen original experiment. It must not forge old
approval hashes, substitute weaker guards, replace saved results, or imply that
unexecuted GPU work passed. This is a tooling gap, not evidence that the saved
experiment was lost. Source checks and CPU reproduction do not certify GPU
behavior in an unrelated runtime.

## Historical versus current documentation

Original scientific source/results/manifests remain intact. Current-facing guide
updates are separately versioned; earlier wrapper manifests and documentation
are copied under `archive/repository-maintenance-20261014-cleanup/` before their
publication checksums are refreshed. The original Task 1 attribution is mapped
to its archived copy by the Task 1 verifier. Do not regenerate historical source
seals merely to make a modified experiment pass.
