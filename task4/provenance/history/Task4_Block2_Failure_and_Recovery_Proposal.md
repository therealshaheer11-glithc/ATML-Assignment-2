# Task 4 Block 2: recorded failure and proposed recovery

## Verified evidence

The user returned the output of environment_attempts/20261011T105114.911388Z. Status: FAILED_PRESERVED. The course installation command, original course environment checker, all released requirement bounds and fresh-process model/data imports passed. Verified versions: Python 3.13.16, Torch 2.11.0+cpu, Transformers 4.57.1, Tokenizers 0.22.1, PEFT 0.17.1, TRL 0.27.2 and NumPy 2.1.3.

The original course function computed 60 audit prompt IDs, 30 SAFE and 30 UNSAFE, and wrote them to the attempt probe result. The final immutable audit and environment contracts were not written because the subsequent global dependency check failed. This is partial environment preparation, not successful final validation.

Global pip check reported exactly three issues:

- IPython 7.34.0: missing Jedi.
- Diffusers 0.40.0: Hugging Face Hub must be >=1.23.0,<2.0; installed 0.36.2.
- Gradio 6.29.0: Hugging Face Hub must be >=1.16.0,<3.0; installed 0.36.2.

Transformers 4.57.1 requires huggingface-hub>=0.34.0,<1.0. Upgrading the shared Hub library to satisfy those Diffusers/Gradio requirements would break that course-pinned requirement. These three distributions are not direct entries in the course requirements. Their exclusion from the complete Task 4 dependency chain has not yet been checked; the recovery must prove it before accepting any exception. No claim is made about whether the missing Jedi predated Block 2.

## Fixed requirements retained

Use the unchanged course requirements and core pins; evaluation only; keep all data/model identities, four frozen policies, deterministic decoding, seed, course helper, audit-selection algorithm and 1,800-response/1,800-judgment/240-human-label workload. Keep the original failed report and source release intact.

## Proposed discretionary change — approval pending

Replace the unconditional whole-Colab dependency gate with a strict Task 4 dependency validation plus a separately reported whole-Colab result. This is a validation-scope decision, not an assignment requirement. No recovery code, package removal, installation or version change has been performed.

If approved, the next single CPU block would:

1. Read back and verify the saved failure report, source identities, original probe result and exact package metadata.
2. Recursively validate the full dependency chain of all released course requirements, respecting installed versions, active environment markers and requested extras. Any missing or incompatible dependency in that chain fails the recovery.
3. Prove that IPython, Diffusers and Gradio are outside that chain. Re-run global pip check, preserve its complete failed result, and permit only the three exact reported issues above. Any additional/different issue stops for review.
4. Re-run fresh-process imports and the unchanged audit selector. Require exact agreement with the already saved 60 IDs rather than silently choosing a different subset.
5. Only after these checks succeed, finalize immutable audit and environment contracts that explicitly distinguish Task 4 dependency success from the still-failed global Colab check and identify the approved exceptions. Preserve commands, logs, versions, hashes, approval text and recovery report in Drive.

This approach needs no new package versions or isolated environment. An isolated environment is an alternative if the user requires a clean whole-environment check; it would need its own reviewed installation/build plan and additional downloads. No alternative is authorized by silence.

## Progress

Inputs: 4/4 verified. Generation: 0/1,800. AI judging: 0/1,800. Human labels: 0/240. Audit selection: 60 IDs computed; final contract pending. CPU environment: core checks passed; global dependency check failed; recovery awaits explicit approval. Remain on CPU.

## Sources

- Released local requirements: task4/course_reference/requirements.txt (preserved course reference).
- Transformers v4.57.1 dependency table: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/dependency_versions_table.py
- pip check semantics: https://pip.pypa.io/en/stable/cli/pip_check/
- User returned output: block2_returned_output.txt (SHA-256 recorded in block2_failed_observation.json).
