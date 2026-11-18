"""Run CMSA-Net without reading GT mask bytes during inference.

Saved GT masks are used only for post-inference metric scoring. This is a
new output-resize protocol; do not directly inherit old comparison scores.
"""
from scripts.run_cmsa_image_only import main

if __name__ == "__main__":
    main()
