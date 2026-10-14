# Repository layout and documentation maintenance

The root README is the current entry point. `docs/PUBLICATION_STATUS.json` records
the completed experiment publication and separates it from report/submission work.
Tasks 1–5 retain isolated execution sources, fixed IDs, saved evidence and CPU
verifiers. This cleanup covers documentation and reproduction instructions only.

Task 1 uses root common/config helpers. Tasks 2–5 preserve the helper versions
used in their executed releases. The root `task5_feedback/` is retained course
scaffolding; the completed implementation is `task5/source/frozen_release/`.

The cleanup adds a CPU saved-results dispatcher and current documentation. Before
updating guide text or publication-wrapper checksums, it preserves their exact
previous bytes in `archive/repository-maintenance-20261014-cleanup/`. Original
experimental source, data identities, results, labels, configs, scientific seals
and release ZIPs are unchanged. The original Task 1 manifest verifies its earlier
attribution page through an explicit archive mapping.

Publication-wrapper manifests track the current guides. They are distinct from
immutable experiment/source manifests. The combined verifier checks both layers.
Raw downloaded datasets, full model weights, caches, and outer transfer ZIPs stay
outside Git. See REPRODUCTION_SCOPE.md for the fresh GPU reproduction boundary.
