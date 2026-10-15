# GPU-trial validation correction

The diagnostic confirmed four original empty rejected answers in standard
training, at zero-based indices 265, 543, 589, and 1381. The original first
600 rows contain three. No other dataset/side has an empty answer.
Each affected answer had zero content tokens before and after preprocessing.
No answer was erased or shortened; released data hashes still match.

The earlier trial stopped before model loading, with zero optimizer updates
and zero GPU allocation. Its prohibition on original empty answers was an
overstrict diagnostic assumption, not an assignment requirement.

Correct only that validation. Verify the diagnostic hashes, original row IDs,
and empty source strings; continue requiring zero prompt/answer truncation.
An empty answer retains one EOS token, scored as log P(EOS | full prompt).
This follows the course EOS convention and approved retain-all-pairs scope.

All datasets, subsets, pair counts, hyperparameters, precision/scaler settings,
optimizer settings, budgets, and stress-selection rules remain unchanged.
Archive the script before and after correction and preserve the failed trial.
Retry one disposable update in a fresh subprocess.
Official training and final configuration remain pending review.
The reward-scoring cap remains 1024 and still needs separate review.
