# Task 4 Block 8: CPU preflight for an isolated environment

## Authorization and entered state

The user declined accepting unresolved GPU dependency exceptions and asked whether the disputed tools were needed. We clarified that Hugging Face Hub is required by the course stack, IPython/Jedi support the surrounding notebook environment, and the supplied Task 4 scripts do not directly use Diffusers or Gradio. The user then replied “sure” to reviewing a setup that preserves Colab's tools and isolates the course packages before making changes.

This release implements only that read-only planning step. It does not implement or execute package repair, installation, removal, environment creation, package-wheel downloads or GPU inference. No GPU exceptions are accepted. Proposed package versions, restoration/removal targets, Drive storage and subsequent installation stages require review and explicit approval after this block returns its concrete result.

The last returned result is GPU_ENVIRONMENT_REVIEW_REQUIRED from gpu_environment_attempts/20261011T142511.766769Z. Fresh imports, the complete task-required graph, A100 float16 CUDA/NF4 checks and all 37 synthetic checks passed. The global check reported IPython missing Jedi plus Diffusers/Gradio versus the course-compatible Hub version. CPU preparation remains complete. No pretrained model has been loaded and no actual evaluation units have been produced.

## Run on CPU

Switch Colab to hardware accelerator None / CPU before running the single launcher. A runtime change can reset temporary files and installed packages. All fixed model snapshots, adapters, source releases and prior reports used here are on Drive. The script reads the saved A100 evidence rather than treating the new CPU software inventory as the target GPU environment.

The launcher mounts Drive, checks the ZIP SHA-256 and every manifested source file, and starts inspect_isolation.py in a separate process. Source and timestamped planning artifacts are saved to Drive. Package contents and earlier source releases are read-only throughout.

## What the preflight establishes

1. Verify the saved Block 7 report and every recorded artifact hash, its sealed source, the CPU orchestration/model-input links and Task 4 ownership. Read the exact original A100 before/after inventories and recorded package changes.
2. Recompute the complete saved GPU dependency graph under its recorded marker environment. Directly inspect whether IPython, Diffusers, Gradio and Jedi occur in that graph. Preserve the measured graph count instead of assuming the CPU count of 64.
3. Propose a fully isolated Python environment, without access to system site-packages, containing exactly the previously verified GPU task-required versions. Preserve Torch 2.11.0+cu130 and the course's fixed Transformers/Tokenizers/TRL/PEFT versions. Include the recorded original pip version as a reproducible installation tool. No latest Torch build or CPU substitute is selected.
4. Query public PyPI release metadata and the official PyTorch CUDA 13.0 wheel index. Select a proposed binary wheel for each exact version, with published SHA-256, filename, URL and size. Compatibility uses this CPU runtime's Python/Linux wheel tags after matching its Python minor version, implementation, OS and architecture to the saved GPU environment. Fresh A100 tag validation remains required before installation. Missing wheels, changed identities, source-only distributions or unexpected versions stop a complete proposal; no substitute version or source build is chosen.
5. Read only the HTTP headers for the exact CUDA Torch wheel's size. No wheel response body is downloaded. Other package sizes and identities come from metadata JSON. Metadata bodies are size-limited, preserved and hashed; no model repository is accessed.
6. Propose restoring only the four recorded global package versions if they are still modified: Hugging Face Hub, Transformers, Tokenizers and PEFT. Propose removing only the two packages newly added globally by the course installation (TRL and bitsandbytes), and only after the isolated copies are installed and fully validated. If a new Colab base already matches the original inventory, those restorations/removals are unnecessary. Unexpected drift requires review. Colab's original IPython, Diffusers and Gradio versions are preserved.
7. Inspect IPython's published Jedi requirement. Propose the latest published stable Jedi candidate satisfying that declared bound, and its Parso dependency. Prefer the original Parso version if it satisfies the bound; otherwise expose a concrete proposed change for approval. These are candidates for the base Colab environment, not new course dependencies or already approved installations. Fresh imports/completion and a clean full base dependency check remain mandatory after any approved repair.
8. Save exact proposed task package pins, all wheel identities, the repair targets, blockers and known compressed download size. Check that this CPU package inventory remains unchanged.

The compatible-wheel selection does not yet prove that a new environment is installed, works or passes dependencies. Exact wheel METADATA and the complete pinned dependency closure must be inspected during an approved CPU wheel-download stage, before A100 installation. No runtime benchmark or installed-size measurement is fabricated.

## Proposed stages to review after the output

- Approve the exact package proposal and Drive wheel storage, using this block's measured compressed download total.
- CPU: download only the approved package wheels into a checksummed Drive wheelhouse. Inspect their real METADATA and uncompressed ZIP entry sizes before GPU installation. The proposed cache uses extra Drive storage and is not created by this block.
- A100: create the isolated environment on local runtime disk, validate wheel compatibility, install offline from approved artifacts and check clean full dependencies, original course imports/helpers, CUDA/NF4 and the 37 synthetic checks.
- Repair the Colab base only within the reviewed restoration/Jedi scope. Check clean full base dependencies and fresh IPython/Jedi/Gradio/Diffusers imports. Keep the notebook launcher in Colab's base environment; run Task 4 scripts through the isolated interpreter. Restart the notebook kernel if needed for the base package changes, then revalidate both environments.
- Continue actual policy and judge loading validation under the original evaluation approval. Model/config/data identities and all fixed evaluation settings remain unchanged.

A virtual environment is local and disposable; recreate it from the approved pins/wheels after a runtime reset. Preserve source, wheel receipts, environment contracts, attempt reports and actual evaluation units on Drive. Do not copy or relocate a virtual environment as a reproducibility mechanism.

## Output and success meaning

- isolation_preflight_releases/<source manifest SHA-256>: this sealed release.
- isolation_preflight_attempts/<UTC timestamp>: START.json, current CPU inventory/platform, recomputed saved GPU graph, membership checks, proposed pins, raw public metadata, request receipts, ISOLATION_PROPOSAL.json and REPORT.json.
- ISOLATION_PREFLIGHT_READY_FOR_REVIEW: all exact proposed wheels were found and their compressed sizes are known. This is a reviewable proposal, not implementation approval or an installed environment.
- ISOLATION_PREFLIGHT_NEEDS_REVIEW: proposal preserves identified artifact blockers; the reported known-size total is incomplete.
- ISOLATION_PREFLIGHT_FAILED_PRESERVED: another catchable failure; available evidence is saved. No automatic repair/retry occurs.

Eight local checks cover wheel name/version/tag/Python bounds, published identities, preservation of the exact CUDA Torch build and narrow restoration boundaries. Syntax is checked locally. Actual Drive/public-metadata validation remains pending until Colab runs this block.

Remain on CPU and return the complete output before downloading wheels or changing packages. Progress remains responses 0/1,800, judgments 0/1,800 and independent human labels 0/240. No training or experimental setting change is proposed. All new code and documentation were prepared with Codex assistance.

## Primary references consulted

- Python 3.13 venv: https://docs.python.org/3.13/library/venv.html — independent package sets, explicit interpreter invocation and recreation after reset.
- pip install: https://pip.pypa.io/en/stable/cli/pip_install/ — requirements, binary artifacts and installation behavior.
- Official CUDA 13.0 Torch wheel index: https://download.pytorch.org/whl/cu130/torch/ — exact CUDA builds; availability and header size still checked by this block.
- Published IPython 7.34.0 metadata: https://pypi.org/pypi/ipython/7.34.0/json — declared Jedi requirement; the actual saved base version is rechecked before planning its repair.
