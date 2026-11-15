#!/usr/bin/env python3
"""Draw the current refinement prototype and the proposed Lean integration.

Usage: .venv/bin/python scripts/draw_contrast_method_figures.py --output-dir '/path/on/Drive/figures_v2'
The plots deliberately identify implemented and proposed components separately.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
if args.output_dir.exists():raise FileExistsError('Use a fresh figure directory')
args.output_dir.mkdir(parents=True)
COLORS={'existing':'#e4f1fb','new':'#fff2cb','mask':'#eee8f8','proposed':'#e4f5e7'}
def box(ax,x,y,w,h,text,kind='existing',fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',fc=COLORS[kind],ec='#263b53',lw=1.0))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#142b43')
def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','color':'#344b61','lw':1.25,'mutation_scale':12})
def canvas(title):
    fig,ax=plt.subplots(figsize=(15,6.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.02,.95,title,fontsize=16,fontweight='bold',color='#142b43');return fig,ax
def save(fig,name):fig.savefig(args.output_dir/name,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)

fig,ax=canvas('A  |  Proposed integration in automatic Lean-SAM2 video propagation')
box(ax,.025,.57,.13,.13,'Frame t\nRGB image');box(ax,.195,.57,.15,.13,'SAM2 image\nencoder');box(ax,.395,.57,.15,.13,'Lean memory\nattention');box(ax,.595,.57,.14,.13,'SAM2 mask\ndecoder');box(ax,.775,.57,.18,.13,'Prototype refiner\nnot integrated', 'new')
for x in (.155,.345,.545,.735):arrow(ax,(x,.635),(x+.04,.635))
box(ax,.025,.79,.145,.085,'One-shot original YOLO\ninitialization','mask',9);arrow(ax,(.17,.83),(.595,.70))
box(ax,.395,.30,.15,.13,'Memory bank\nexisting pruning');box(ax,.595,.30,.14,.13,'Memory encoder\nexisting')
arrow(ax,(.47,.43),(.47,.57));arrow(ax,(.595,.365),(.545,.365));arrow(ax,(.865,.57),(.865,.43));arrow(ax,(.865,.43),(.735,.365))
arrow(ax,(.27,.57),(.27,.16));arrow(ax,(.27,.16),(.595,.30))
ax.text(.025,.10,'The refiner additionally receives early image features and the memory-conditioned deep feature (see B).',fontsize=10,color='#42576b')
ax.text(.025,.06,'Initial conditioning is protected. Integration and propagation remain proposed; the refiner and routing function exist.',fontsize=10,color='#42576b')
ax.text(.025,.02,'No new detector call, confidence gate or token-pruning rule is proposed.',fontsize=10,color='#42576b');save(fig,'pipeline.png')

fig,ax=canvas('B  |  Standalone refinement prototype and proposed factorial routing')
box(ax,.025,.73,.19,.105,'Early FPN feature\n32 × 256 × 256');box(ax,.025,.56,.19,.105,'Memory-conditioned\ndeep feature\n256 × 64 × 64');box(ax,.025,.39,.19,.105,'Selected original logits\n1 × 256 × 256','mask')
box(ax,.275,.66,.17,.15,'1×1 projections +\nbilinear alignment','new');arrow(ax,(.215,.78),(.275,.76));arrow(ax,(.215,.61),(.275,.70))
box(ax,.50,.73,.19,.115,'TCMN-derived CBAM\nchannel reweighting','new',9);arrow(ax,(.445,.75),(.50,.79))
box(ax,.50,.49,.19,.135,'Soft mask-weighted\nforeground/background\ncosine contrast','new',9);arrow(ax,(.445,.68),(.50,.57));arrow(ax,(.215,.44),(.50,.51))
box(ax,.755,.60,.20,.14,'Residual head + logits\nzero-initialized head','new');arrow(ax,(.69,.79),(.755,.70));arrow(ax,(.69,.56),(.755,.64))
arrow(ax,(.215,.44),(.72,.44));arrow(ax,(.72,.44),(.72,.61));arrow(ax,(.72,.61),(.755,.61))
box(ax,.29,.13,.67,.16,'Routing under test: A original / original     B refined / original\nC original / refined     D refined / refined\n(first = displayed logits; second = memory-write logits)','proposed',10);arrow(ax,(.855,.60),(.855,.29))
ax.text(.025,.035,'The gate reweights channels; it is not full cross-channel value mixing. No trained checkpoint or video result is implied.',fontsize=9.4,color='#42576b')
save(fig,'architecture.png')
print(f'Wrote {args.output_dir / "pipeline.png"} and {args.output_dir / "architecture.png"}')
