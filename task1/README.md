# Task 1 — Direct Preference Optimization

Start with [results](docs/RESULTS.md), [protocol and decisions](docs/PROTOCOL.md),
and the [evidence audit](docs/AUDIT.md).

| Folder | Contents |
|---|---|
| `source/` | Original implementation, helper modules, configs, dependency pins, preparation scripts and tests |
| `results/` | Saved experiment evidence and analysis tables |
| `figures/` | Published figures and their provenance |
| `docs/` | Current result, protocol and reproduction guides |

From the repository root, reproduce the saved statistics on CPU:

```bash
python3 -B task1/reproduce.py --output outputs/task1_recomputed
```

To rebuild figures with Matplotlib installed:

```bash
python3 -B task1/plot_results.py --output outputs/task1_figures
```

For new model execution, use the exact commands in the [root README](../README.md#task-1-standard-dpo-beta-sweep-and-length-control),
starting from `task1/source/`. The experimental implementation and configurations have moved without edits. One publication test now points to the relocated evidence.
Use [reproduction guidance](docs/REPRODUCE.md) for the original settings and inputs.
Historical manifests and paths are verified through the [layout map](../manifests/README.md).
