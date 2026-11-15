# External implementations

The project uses pinned author repositories as Git submodules. The primary
comparators are `sam2`, `Lean-SAM2`, `SAM2Long`, `SAMURAI` and `CMSA-Net`.
Additional registered submodules support older studies or comparisons; their
presence does not make them part of the current method.

Initialize registered checkouts with `git submodule update --init --recursive`.
The three `Lean-SAM2-*-window*`/`Lean-SAM2-raw-two-control` variants are pinned
to the same author revision, with project-specific changes recorded separately
in `patches/`. See [the patch guide](../patches/README.md) before running them.

Other local author checkouts may exist in this directory for historical
experiments. They are ignored unless explicitly registered as submodules; they
are not automatically published or required by every workflow. Consult the
corresponding experiment protocol for its exact dependencies and revisions.
