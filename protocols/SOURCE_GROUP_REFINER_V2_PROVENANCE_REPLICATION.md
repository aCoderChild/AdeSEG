# Source-group refiner v2 provenance replication

The v1 fixed-read study completed and its saved-mask arithmetic was independently
checked. Its loader, however, reused the local variable `path` in a later loop
and returned the final capture receipt hash as `manifest_sha256`. The loader
also checked the actual manifest against its completion receipt, but its plan,
checkpoints and scorer identify the wrong file under that name. The earlier
run is preserved and must not be presented as correctly manifest-bound.

The v2 copies retain the same videos, sample selection, model, four arms,
training seed and objective. They fix the loader to return the actual manifest
hash and bind v2 sources in the study plan. The earlier masks and labels have
already been inspected; this is a provenance-corrected replication, not a
fresh blinded validation or a separate independent experiment. The earlier
v1 source and evidence remain unchanged.

As in the v1 protocol, the nine audited MPS captures, real-data CPU/MPS parity,
matched training, frozen validation masks and independent no-label replay are
prerequisites to scoring. The source-group split does not establish patient
independence. Even successful fixed-read scores do not establish video feedback
efficacy or safety. The launcher uses a fresh Drive output directory, checks
22 GiB free before each phase and logs space during phases without interrupting
an active phase.
