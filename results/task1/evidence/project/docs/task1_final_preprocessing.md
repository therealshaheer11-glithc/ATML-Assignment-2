# Final Task 1 preprocessing and evaluation limits

The user explicitly approved all three 4096-token limits after the disposable
GPU trial passed. The generation ceiling remains 256 new tokens; all other
approved training and decoding settings remain unchanged.

Original course starter: 05543bec4fba95eba708eae9d6e15730d53ca317.
TA preprocessing patch: 076e7cfdc7cd2f4b5e8e9e12cf15ef3652bc92da.
Only encode_prompt_response changes in the upstream common/data.py patch.
It keeps the full prompt, reserves EOS, and truncates the answer prefix if
necessary. A prompt at or above the cap raises rather than silently disappearing.
The earlier 4096-token audit found no answer truncation in our exact released
datasets. No pairs are removed or replaced; the short pool is the first 600
original standard rows before shuffling. The four original empty rejected
answers remain valid EOS-only responses, scored as immediate stopping.

Two historical tests described the old prompt-truncation behavior. They are
archived, updated for the TA rule, and supplemented with empty-answer and
oversized-prompt checks. The objective and causal scoring helpers are unchanged.
Strict accuracy uses margin > 0; ties count as zero wins in the full denominator.

The pinned model revisions are declared in configs/dpo.yaml. Completed Task 1
loaders must honor those fields; the unfinished original loaders do not yet.
Likewise, generation and reward scripts must explicitly pass the configured
limits to helpers. Their old defaults must not be used for official Task 1.
No model weights are loaded and no official training is started by this cell.
Full generated reward inputs and reward-model GPU memory still need validation.
