# Training-only output-fidelity control

Exploratory development experiment, defined before running this follow-up. The
30-epoch multilevel refiner reduced training loss but degraded previously
inspected validation Dice. This tests whether discouraging large deviations
from the original prediction reduces overcorrection; it does not presume a
benefit for unseen videos or false alarms.

Use the unchanged `MultilevelContrastRefiner` and existing 17/10 capture
manifest. Train an otherwise identical multilevel refiner and mask-only control
for 30 epochs with three seeds: 20261114, 20261115, 20261116. Adam learning
rate 0.0003, weight decay 0.0001, existing BCE-plus-soft-Dice objective and
data order. Add `0.1 * mean((sigmoid(refined) - sigmoid(original))**2)` to
the training loss; the original logits are detached. The loss uses prediction
and label as existing training inputs, with no new inference threshold, gate,
prompt, detection, or pruning mechanism. The penalty applies to all training
examples, including labeled-empty examples, so it may also prevent necessary
false-alarm correction. This is a concrete tradeoff rather than a safety claim.

Train seq2/3/4: 15 original foreground reads plus two label-selected seq4
empty false alarms. Validate on all ten previously inspected seq9/10 reads;
freeze checkpoint and all prediction files before reading their labels. Report
each seed, baseline and mask-only control, present-frame Dice, empty-frame
false-alarm counts and predicted area, frame outcomes and checkpoint/source
hashes. Preserve the prior 5- and 30-epoch results as separate comparisons.
This previously reused data cannot support independent validation, and its
absence of no-object-logit cases must be reported. No full-video efficacy run
is authorized by this pilot alone.

Run CPU/MPS fixed-read loss and gradient checks on a present and a labeled-empty
training example before MPS training. Results, logs and checkpoints go to the
configured Google Drive experiment directory; do not mix these GPU pilots with
unmatched video baselines.
