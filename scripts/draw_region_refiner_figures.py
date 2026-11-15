#!/usr/bin/env python3
"""Draw the implemented region-conditioned prototype and proposed study.

The diagrams describe code and routing, not evidence of a trained-video gain.
Use --output-dir to write a fresh experiment folder on the user's Drive.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    colors = {'model': '#e4f1fb', 'new': '#fff2cb', 'prompt': '#eee8f8', 'route': '#e4f5e7'}

    def canvas(title):
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        ax.text(.02, .96, title, fontsize=16, fontweight='bold', color='#142b43')
        return fig, ax

    def box(ax, x, y, w, h, label, kind='model', size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.007,rounding_size=0.012',
                                    fc=colors[kind], ec='#263b53', lw=1.1))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=size, color='#142b43')

    def arrow(ax, start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle':'-|>', 'color':'#344b61', 'lw':1.25, 'mutation_scale':12})

    def save(fig, name):
        fig.savefig(args.output_dir/name, dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, ax = canvas('A  |  Proposed matched automatic-video experiment')
    box(ax,.025,.62,.13,.12,'Frame t\n(and seq23 crop)')
    box(ax,.19,.62,.15,.12,'SAM2 image\nencoder')
    box(ax,.385,.62,.15,.12,'Lean memory\nattention')
    box(ax,.58,.62,.14,.12,'SAM2 mask\ndecoder')
    box(ax,.765,.62,.2,.12,'Region-conditioned\nrefiner prototype','new')
    for x in (.155,.34,.535,.72): arrow(ax,(x,.68),(x+.045,.68))
    box(ax,.025,.82,.19,.075,'Archived automatic mask\none initialization','prompt',9)
    arrow(ax,(.215,.855),(.58,.74))
    box(ax,.385,.34,.15,.12,'Existing memory\nselection & bank')
    box(ax,.58,.34,.14,.12,'Native memory\nencoder')
    arrow(ax,(.46,.46),(.46,.62)); arrow(ax,(.58,.4),(.535,.4))
    arrow(ax,(.865,.62),(.865,.51)); arrow(ax,(.865,.51),(.72,.4))
    box(ax,.765,.21,.2,.11,'A/B/C/D: output and\nmemory-write routing','route',9)
    arrow(ax,(.865,.51),(.865,.32))
    ax.text(.025,.12,'Early image features and memory-conditioned deep features also enter the refiner.',fontsize=10,color='#42576b')
    ax.text(.025,.07,'The four-arm runner is development-only; no trained region-conditioned video result is established.',fontsize=10,color='#42576b')
    ax.text(.025,.025,'No new detector, confidence gate or token-pruning rule is introduced by the proposed refiner.',fontsize=10,color='#42576b')
    save(fig,'pipeline.png')

    fig, ax = canvas('B  |  Predicted-region-conditioned channel reweighting')
    box(ax,.025,.73,.16,.105,'Early FPN\n32 × 256 × 256')
    box(ax,.025,.55,.16,.105,'Memory-conditioned\ndeep feature\n256 × 64 × 64')
    box(ax,.025,.34,.16,.105,'Original mask logits\n1 × 256 × 256','prompt')
    box(ax,.24,.64,.16,.14,'1×1 projections\nand spatial alignment','new')
    arrow(ax,(.185,.78),(.24,.75)); arrow(ax,(.185,.60),(.24,.68))
    box(ax,.445,.70,.17,.12,'Soft foreground and\nbackground feature means','new',9)
    arrow(ax,(.4,.72),(.445,.76)); arrow(ax,(.185,.39),(.425,.39)); arrow(ax,(.425,.39),(.445,.70))
    box(ax,.66,.70,.17,.12,'TCMN-derived CBAM\nshared channel MLP','new',9)
    arrow(ax,(.615,.76),(.66,.76))
    box(ax,.66,.49,.17,.12,'Channel weights ×\nfused spatial features','new',9)
    arrow(ax,(.745,.70),(.745,.61)); arrow(ax,(.4,.66),(.66,.55))
    box(ax,.445,.32,.17,.12,'Foreground–background\ncosine contrast','new',9)
    arrow(ax,(.53,.70),(.53,.44)); arrow(ax,(.4,.64),(.445,.39))
    box(ax,.865,.43,.11,.19,'Residual head\n+ original\nlogits','new',9)
    arrow(ax,(.83,.55),(.865,.54)); arrow(ax,(.615,.38),(.865,.46))
    arrow(ax,(.185,.39),(.2,.18)); arrow(ax,(.2,.18),(.92,.18)); arrow(ax,(.92,.18),(.92,.43))
    ax.text(.025,.085,'The shared MLP combines channel information to calculate weights; it does not mix output channel values.',fontsize=9.6,color='#42576b')
    ax.text(.025,.04,'The original mask can be suppressed to −1024 by SAM2; preserving its presence decision may limit recovery.',fontsize=9.6,color='#42576b')
    save(fig,'architecture.png')
    print(f'Wrote {args.output_dir / "pipeline.png"} and {args.output_dir / "architecture.png"}')


if __name__ == '__main__':
    main()
