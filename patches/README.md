# Author-code adaptations

Patches preserve local modifications separately from pinned author checkouts.
Apply only the patch required by the experiment, after initializing submodules.
From the project root, check first and then apply:

```sh
git -C external/Lean-SAM2-token-window apply --check ../../patches/lean-sam2-token-window.patch
git -C external/Lean-SAM2-token-window apply ../../patches/lean-sam2-token-window.patch
```

The Lean variants all use upstream commit
`8d1cf2618519d291905d22fe859b8302476d7f50`:

| Checkout | Patch | Purpose |
| --- | --- | --- |
| `Lean-SAM2-token-window` | `lean-sam2-token-window.patch` | Window runtime used by the refiner studies |
| `Lean-SAM2-fused-window-control` | `lean-sam2-fused-window-control.patch` | Historical fusion control |
| `Lean-SAM2-raw-two-control` | `lean-sam2-raw-two-control.patch` | Historical raw control |

Replace both paths in the command above to apply a different patch. The SASVi
adaptation is in `sasvi_confirmation.patch`; the Efficient-SAM2 adaptation is
documented in [efficient-sam2-device.md](efficient-sam2-device.md). A patch
must not be applied twice: `git apply --check` fails when it is already applied.
