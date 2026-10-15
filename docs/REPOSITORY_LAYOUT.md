# Repository layout

| Location | Purpose |
|---|---|
| [task1/](../task1/) | DPO source, results, figures and guides |
| [task2/](../task2/) | PPO source, results, audits and launchers |
| [task3/](../task3/) | GRPO source, results, audits and launchers |
| [task4/](../task4/) | Safety execution sources, evidence and analysis |
| [task5/](../task5/) | Feedback evaluation source, results and analysis |
| [scripts/](../scripts/) | Repository-wide verification and CPU reproduction |
| [manifests/](../manifests/) | Current checksums and original-to-current path records |
| [archive/](../archive/) | Earlier source scaffolds and superseded publication files |

The root README supplies exact entry points. Each task has its own README.
Task 1 model commands run from `task1/source/`; its original import/config/data
relationships remain intact. Public saved-result commands run from the repository root.

The 15 October structural cleanup moves source/evidence without changing their
bytes. Any edited publication guide or verifier has its previous bytes retained
under `archive/repository-layout-20261015/`. A baseline map accounts for every
file at commit `2cabffbec6e72fbb4f865b72af485e10c9c96579`.
Original source/experiment manifests are not regenerated. Current publication
wrapper manifests are refreshed only for the changed navigation or paths.

Historical paths inside frozen logs, notebooks and source documentation identify
the original execution environment. They are not instructions to create duplicate
root folders. See [the map documentation](../manifests/README.md).

This reorganization changes neither the experiment results nor the documented
[fresh-runtime reproduction limits](REPRODUCTION_SCOPE.md). Task 6 and the final
course report remain separate work.
