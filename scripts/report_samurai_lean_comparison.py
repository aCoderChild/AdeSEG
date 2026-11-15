#!/usr/bin/env python3
"""Report historical matched-frame SAMURAI, native SAM2 and Lean results."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment');dirs={'native':p/'08_region_readout_polypgen_v1/serial_release_v1/sam2_full','lean':p/'lean_polypgen_full_v1/attempt03/full','samurai':p/'12_author_region_attention_polypgen_cpu_v1/samurai_full'};seqs=[f'seq{i}' for i in range(2,24) if i!=7];read=lambda x:json.loads(x.read_text());setup={k:read(v/'setup.json') for k,v in dirs.items()}
for name in dirs:
 if not read(dirs[name]/'delivery.json')['completed']:raise ValueError('Run incomplete')
 for key in ('checkpoint_sha256','manifest_sha256','device','torch','precision','apply_postprocessing','memory_stride'):
  if setup[name][key]!=setup['native'][key]:raise ValueError(f'{name} differs on {key}')
 if setup[name]['initialization']!='first_labeled_frame':raise ValueError('Prompt differs')
rows=[]
for seq in seqs:
 records={name:[json.loads(x) for x in (folder/seq/'frames.jsonl').read_text().splitlines()] for name,folder in dirs.items()};ids={k:[r['source_frame_index'] for r in v] for k,v in records.items()}
 if any(ids[k]!=ids['native'] for k in ids):raise ValueError(f'{seq} frame identities differ')
 if any([r['present'] for r in records[k]]!=[r['present'] for r in records['native']] for k in records):raise ValueError('Presence differs')
 stats={}
 for name,frames in records.items():
  present=[f['scores']['dice'] for f in frames if f['present']];empty=[f for f in frames if not f['present']]
  stats[name]={'present_dice':float(np.mean(present)),'present_count':len(present),'empty_count':len(empty),'false_alarms':sum(f['absent_fp'] is True for f in empty)}
 rows.append({'sequence':seq,'target_count':len(ids['native']),'methods':stats,'frame_records_sha256':{k:digest(folder/seq/'frames.jsonl') for k,folder in dirs.items()}})
summary={k:{'equal_sequence_present_dice':float(np.mean([r['methods'][k]['present_dice'] for r in rows])),'false_alarms':sum(r['methods'][k]['false_alarms'] for r in rows),'empty_count':sum(r['methods'][k]['empty_count'] for r in rows)} for k in dirs}
out=validate_output(p/'lean_experiment_evidence_v1/samurai_comparison_attempt01');out.mkdir(parents=True,exist_ok=False)
result={'rows':rows,'summary':summary,'setups_sha256':{k:digest(v/'setup.json') for k,v in dirs.items()},'runner_sha256':digest(Path(__file__)),'scope':'Historical GT-initialized matching source-frame records; seq23 original full input.'};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# SAMURAI alongside native SAM2 and Lean-SAM2','','Historical runs use one ground-truth initialization, the same checkpoint and exactly matching post-prompt source-frame IDs. SAMURAI uses its author configuration and re-bases the video at the prompt, so video handling differs. All runs are exploratory CPU FP32; the older Lean CPU RoPE issue and previously inspected dataset limit inference. Seq23 is the original full-frame input, including the inset.','','| Method | Equal-sequence present Dice | Empty-frame false alarms |','|---|---:|---:|']
for k in ('native','lean','samurai'):
 v=summary[k];lines.append(f"| {k} | {v['equal_sequence_present_dice']:.4f} | {v['false_alarms']} / {v['empty_count']} |")
lines+=['','| Priority sequence | Native Dice | Lean Dice | SAMURAI Dice | SAMURAI empty FAs |','|---|---:|---:|---:|---:|']
for row in rows:
 if row['sequence'] not in ('seq8','seq11','seq20','seq23'):continue
 m=row['methods'];lines.append(f"| {row['sequence']} | {m['native']['present_dice']:.4f} | {m['lean']['present_dice']:.4f} | {m['samurai']['present_dice']:.4f} | {m['samurai']['false_alarms']} / {m['samurai']['empty_count']} |")
lines+=['','These historical scores are **not matched automatic-prompt comparisons** against the corrected YOLO-prompted Lean experiments. In particular, the latter crop seq23 and use the later indexed CPU RoPE implementation. The difference cannot be attributed to SAMURAI memory alone, because its full tracking system and video handling differ. GT-empty labels may be incomplete, detector training exposure and patient independence are unknown. Frame IDs and presence were checked here; this report reuses recorded metrics and is not an independent mask rescore. Full per-sequence results and source hashes are in results.json.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
