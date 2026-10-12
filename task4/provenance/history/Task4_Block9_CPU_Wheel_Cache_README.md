# Task 4 Block 9: approved CPU package cache

The user replied “approved” to the exact isolated setup proposal documented in APPROVED_PROPOSAL.md. APPROVAL.json records that authorization. This release implements the next CPU stage only: download, checksum, resume, metadata/dependency inspection and size measurement. A100 environment creation and conditional Colab base repair are approved later stages, not operations of this block.

## Entered state

Block 8 returned ISOLATION_PREFLIGHT_READY_FOR_REVIEW from isolation_preflight_attempts/20261011T145304.669615Z. Its saved A100 chain contains 83 required distributions; the reviewed isolated environment adds pip for 84 pins. The complete proposal has 90 binary wheels totaling 2,969,739,166 compressed bytes (2.766 GiB), with no unresolved candidates. The other six wheels cover the four original Colab package versions, Jedi 0.20.0 and the existing compatible Parso version. The exact 84 versions printed in that output are frozen in APPROVED_ISOLATED_VERSIONS.json. No package versions are selected during download.

The underlying A100 result remains GPU_ENVIRONMENT_REVIEW_REQUIRED. No GPU exception is approved and no GPU environment contract is finalized. No pretrained model has been loaded. Task 4 responses 0/1,800; judgments 0/1,800; independent human labels 0/240.

## Run exactly one block on Colab CPU

Keep hardware accelerator None / CPU. Download ATML_PA2_Task4_CPU_Wheel_Cache.zip and run the single supplied launcher; upload only that ZIP when prompted. The launcher verifies the archive and source manifest and streams the CPU worker's output. It does not install anything on the user's computer or in Colab.

The worker checks Task 4 ownership, all referenced source identities, the exact saved Block 8 and Block 7 reports/artifact hashes, frozen protocol/model-input contract links, and approved versions. It requires the saved A100 Python minor version/implementation, operating system and architecture to match the CPU inspection runtime; fresh A100 wheel tag validation remains necessary later. It validates all 90 filenames, exact versions, compatible tags, URLs, SHA-256 identities and byte sizes before downloading.

No model repository is accessed. Existing model files, original adapter configs, course helpers, frozen requirements, reports and previous source releases are read only. The four exact course pins and Torch 2.11.0+cu130 remain unchanged. The full package set is the approved saved GPU set; no source build, alternate version, CPU Torch substitution, dependency exception, tuning or training is introduced.

## Saved cache and resume

Drive root: `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`.

- `wheel_cache_releases/<source manifest SHA-256>/`: sealed source, approved proposal and approval evidence.
- `wheel_cache_attempts/<UTC attempt>/`: timestamped START/REPORT, before/after CPU package inventories, approved proposal, platform/tags, download-start and per-wheel inspection records, dependency graphs and size measurements.
- `environment_wheelhouse/CACHE_IDENTITY_CONTRACT.json`: immutable binding of the exact files to the approved proposal and source.
- `environment_wheelhouse/wheels/`: complete published package files.
- `environment_wheelhouse/receipts/`: one immutable SHA-256/size receipt per complete file.
- `environment_wheelhouse/incomplete/<UTC attempt>/`: incomplete transfer files, preserved after interruption/failure.
- `environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json`: finalized only after all 90 file and metadata checks plus complete dependency closure pass.

All complete cached files and receipts are checked before any new network request. A missing receipt can be reconstructed only after checking the file against its published approved hash. A corrupt, unexpected or missing receipted file stops the run and is preserved; nothing is automatically repaired or replaced.

Each missing file is streamed into an attempt-specific partial file on Drive, with bounded reads, allowed HTTPS publisher redirects, byte count and SHA-256 checks. The file is read back for verification before publication as a complete cache unit. Receipts are saved after each complete file. Resume is per complete file: a verified complete file is reused; an interrupted file restarts from byte zero under a new attempt, with the earlier partial copy preserved. This is not a byte-range resume. Leftover partial copies consume additional Drive space and are not automatically deleted. Do not run concurrent writers.

## Actual wheel checks and size meaning

Downloaded wheel ZIPs are inspected without extraction or package-code execution. The worker rejects unsafe paths, duplicate members, symlinks/encryption, incompatible filenames/tags/Python bounds, missing wheel format files and changed METADATA package identities. It reads each wheel's actual Requires-Dist, Provides-Extra and Requires-Python fields.

The original preserved dependency inspector is reused to validate markers and requested extras against the saved GPU platform. It checks both the unchanged course roots (the same 83-package version graph) and the entire isolated 84-package set, including pip. Missing, incompatible or newly required dependencies stop the run; there is no automatic additional download or version change. Jedi's actual two-package closure must match the approved Jedi/unchanged-Parso repair and original IPython bound. Full repaired Colab base validation remains a later A100 stage.

The worker reports each wheel's ZIP logical uncompressed bytes, the isolated 84-wheel total and the largest isolated wheel. These are measured from archive entries, not measured installed disk usage. Bytecode, filesystem overhead and installer temporary files require additional space. The later A100 block must inspect actual local disk availability before installation. Download and installation timing are not predicted.

## Success and failures

CPU_WHEEL_CACHE_VERIFIED_NO_INSTALLATION means all 90 package files and their published hashes/sizes pass, all 90 metadata inspections pass, course and full isolated closures pass, Jedi/Parso closure passes, and the CPU package inventory is unchanged. It does not mean either environment has been installed, repaired or validated on A100.

CPU_WHEEL_CACHE_FAILED_PRESERVED means the catchable failure and available evidence are recorded. No automatic retry, repair, installation or removal follows. Complete checked units remain available for resume; read the exact error before proceeding.

Return the entire output after this block. Subsequent A100 and base-repair blocks depend on that result, following the approved one-executable-block-at-a-time cadence. Generation/judging/human progress remains zero until actual evaluation stages.

## Local validation and attribution

Thirteen meaningful synthetic tests cover exact approval/version/count boundaries, metadata/ZIP/tag/Python guards, compatible and incompatible transitive dependency closures including extras, cached receipt/identity validation, bounded network reads, checksum/redirect failures, verified-file publication and interrupted-transfer preservation/resume. They use small synthetic files and mocked responses; they do not download the actual 90 wheels, install packages, create an environment or validate a GPU. LOCAL_VALIDATION.json and LOCAL_TEST_OUTPUT.txt record the actual local results.

The course source and original dependency inspector remain unchanged. New cache orchestration, tests, documentation and packaging were created with Codex assistance under the recorded user approval.
