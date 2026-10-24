# Relapse-prevention feasibility — no inference

**2026-10-24. Option A, pre-inference amendment. Execution remains on user HOLD.**
The prior rule/motivation mismatch is real: a nonempty native mask requires a
positive object score, so previous-score≤0 does not directly interrupt a
continuing positive-score false alarm on the native trajectory. The rule is
retained only to test **relapse prevention after model-declared absence**, with
present-frame and re-detection costs. It is not the propagation equivalent of
the42% one-step foreground-clearance finding.

## Counts from completed native baselines

Each method covers21 promptable sequences,2,012 targets and323 GT-empty targets.
Archived masks were read, not recomputed. The prior GT prompt is nonempty by
construction. Counts use sampled chronological targets, not consecutive source
video frames or seconds.

| Baseline quantity | Native SAM2 | Native SAMURAI |
|---|---:|---:|
| GT-empty targets following an empty prediction | 139 | 95 |
| Of these, previous GT also empty | 131 | 91 |
| Of these, previous GT present (a missed prediction) | 8 | 4 |
| False alarms following an empty prediction | 5 | 9 |
| False alarms following a nonempty prediction | 158 | 208 |
| All GT-empty false alarms | 163 | 217 |
| Consecutive GT-empty targets with positive predictions at both steps | 128 | 166 |
| Present-GT targets following an empty prediction | 143 | 37 |

The139/95 targets are **mask-proxy opportunities**, not exact score triggers.
Legacy object scores were not saved. Score≤0 implies an empty mask; the reverse
does not hold. Positive-score decoders can also produce empty masks. Therefore
actual nonpositive-score opportunities are a subset of these counts.

Only5/163 SAM2 and9/217 SAMURAI false-alarm targets follow an empty prediction.
Direct native-trajectory relapse headroom is limited. However, these are **not
global upper bounds** on effects of full propagation: preventing a relapse may
alter later memory, predictions and eligibility. Do not claim a maximum of5/9
improvable frames, or infer general failure of region-aware methods.

## Consequences for design

- Report relapse prevention, not clearing ongoing persistent false alarms.
- Use a fixed risk set from **unmodified native SAMURAI scores** for the paired
  primary comparison. Conditioning on the gated method's own scores would select
  post-treatment frames and change the question.
- Save prompt/current/previous native scores in any subsequently authorized run.
  Overall false alarms remain descriptive; present Dice and return failures/delay
  are mandatory costs. Zero-opportunity sequences are N/A.
- A gate after a missed detection may delay recovery, but “can only hurt” is not
  mathematically guaranteed: changing nonlinear attention can help or hurt.
- Do not tune a new confidence threshold or fit an absence detector here.
  Persistent-FP targeting requires a separately designed/calibrated strategy.
- No large experiment is automatically queued. Sparse direct headroom limits
  the interpretation of a negative result; this is not a decisive test of the
  broader regional hypothesis or a method-versus-analysis-paper verdict.

## Verification and reproduction

`scripts/audit_relapse_opportunities.py` downloads preserved native author
archives through the existing W&B API; archive MD5 and ZIP CRC checks passed.
An independent read compared metric-derived occupancy with the actual archived
PNGs on **2,012/2,012 SAM2 and2,012/2,012 SAMURAI targets**, with no disagreement.
Per-sequence counts, artifact identities, SHA256 checks and mask verification
are under `outputs/experiment/20_relapse_opportunity_audit_v2/`.
The opportunity report SHA256 is
`9fcba6d5d85d478df84ce56075a06b20c44243bad027a9e1a9236a96687c1b3b`.

The amended local regression suite has **181 passing tests**. The earlier local
freeze recorded171, while the user's environment reported169. No untracked test
files were present at the start of this amendment; that difference cannot be
attributed to local-only tests without comparing collection/environment details.
Unit tests are not real-video gate or full-propagation results.

[Amended protocol](../protocols/full_propagation_false_alarm_v1.json)
· [Implementation and future execution guide](FULL_PROPAGATION_FALSE_ALARM.md).
