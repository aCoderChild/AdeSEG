"""Run171: frozen-history group contribution, reusing Run170's verified path."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust
from modeling.cfbi_training import capture_training_read
from modeling.token_contribution import ARMS, TokenContributionRead
from scripts import run_complementary_memory_pilot as base
from scripts.bank_integrity import bank_contents, fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_replay import check_exact, check_layout
from scripts.run_regional_token_selection import replay, scored, SEQUENCES, EXTRA_SOURCES

PROTOCOL = "protocols/token_contribution_v1.json"
SOURCES = (PROTOCOL, "modeling/token_contribution.py", "scripts/run_token_contribution.py",
           "tests/test_token_contribution.py")


def spaced(values, count):
    return [values[int(i)] for i in np.linspace(0, len(values)-1, min(count, len(values))).round()] if values else []


def target_plan(videos):
    plans = {}
    for name in SEQUENCES:
        frames = videos[name]
        prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
        present, absent = [], []
        for i in range(prompt+3, len(frames)):
            (present if (load_label_mask(frames[i].mask_path, [1]) == 1).any() else absent).append(i)
        plans[name] = {"prompt": prompt, "targets": sorted(spaced(present, 4)+spaced(absent, 2)),
                       "present_eligible": len(present), "absent_eligible": len(absent),
                       "total_frames": len(frames)}
        if not plans[name]["targets"]:
            raise RuntimeError(f"No eligible diagnostic targets for {name}")
    return plans


def summarize(sequences):
    """Paired sequence means; no frame-level pseudo-replication or cherry-pick."""
    by_arm = {arm: {} for arm in ARMS}
    interaction = {}
    for name, rows in sequences.items():
        native = rows["native"]
        if any([r["frame"] for r in rows[a]] != [r["frame"] for r in native] for a in ARMS):
            raise RuntimeError("Unmatched diagnostic targets")
        for arm in ARMS:
            values = rows[arm]
            if any(r["present"] != n["present"] for r, n in zip(values, native)):
                raise RuntimeError("Unmatched GT denominators")
            present = [r for r in values if r["present"]]
            absent = [r for r in values if not r["present"]]
            by_arm[arm][name] = {"present_frames": len(present), "empty_frames": len(absent),
                **{m: float(np.mean([r['scores'][m] for r in present])) if present else None
                   for m in ("dice", "iou", "boundary_f")},
                "false_alarms": sum(r["absent_fp"] for r in absent),
                "false_alarm_rate": float(np.mean([r["absent_fp"] for r in absent])) if absent else None}
        if by_arm['native'][name]['dice'] is not None:
            # U(AB)-U(A)-U(B): positive means joint removal hurts more than additive.
            d = lambda a: by_arm[a][name]['dice']
            interaction[name] = d('rejected_a')+d('rejected_b')-d('rejected')-d('native')
    means = {}
    for arm, seq in by_arm.items():
        means[arm] = {m: float(np.mean([r[m] for r in seq.values() if r[m] is not None]))
                     if any(r[m] is not None for r in seq.values()) else None
                     for m in ('dice', 'iou', 'boundary_f', 'false_alarm_rate')}
        means[arm].update(empty_frames=sum(r['empty_frames'] for r in seq.values()),
                          false_alarms=sum(r['false_alarms'] for r in seq.values()))
    utility, descriptors = {}, {}
    for arm in ARMS[1:]:
        values, descriptions = [], []
        for name, rows in sequences.items():
            for native, altered in zip(rows['native'], rows[arm]):
                if native['present']:
                    values.append(native['scores']['dice']-altered['scores']['dice'])
                for layer in altered.get('read', {}).get('selection_layers', []):
                    for group in layer['intervention_groups']:
                        descriptions.append({**group, 'age':altered['frame']-group['memory_frame']})
        utility[arm] = {'sampled_present_targets':len(values),
            'removal_hurts':sum(v > 0 for v in values), 'removal_helps':sum(v < 0 for v in values),
            'exact_dice_ties':sum(v == 0 for v in values),
            'median_native_minus_removed_dice':float(np.median(values)) if values else None,
            'note':'Target counts are descriptive, not independent statistical observations; exact Dice ties need not mean identical masks.'}
        descriptors[arm] = {field:float(np.mean([d[field] for d in descriptions])) if descriptions else None
            for field in ('removed','retained_fraction','first_read_usage_mass','mean_soft_foreground','mean_vector_norm','age')}
    contrasts, pvalues = {}, {}
    for label, left, right in (("rejected_minus_retained", "rejected", "retained"),
                               ("rejected_minus_random", "rejected", "random")):
        l = {n:r['dice'] for n,r in by_arm[left].items() if r['dice'] is not None}
        r = {n:by_arm[right][n]['dice'] for n in l}
        delta = {n:l[n]-r[n] for n in l}
        contrasts[label] = {"sequence_deltas": delta, "mean": float(np.mean(list(delta.values()))),
            "bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=20261107),
            "wins": sum(v > 0 for v in delta.values()), "n": len(delta)}
        pvalues[label] = paired_wilcoxon(l,r)['p']
    full = len(sequences) == len(SEQUENCES)
    signal = full and all(c['bootstrap_95_ci'][0] > 0 and c['wins'] > c['n']/2 for c in contrasts.values())
    return {"scope": "fixed-native-history group diagnostic; not individual-token or propagation efficacy",
            "complete_sequences": len(sequences), "per_sequence": by_arm, "means": means,
            "target_utility_distribution":utility,"removed_group_descriptors":descriptors,
            "primary_contrasts": contrasts, "holm_pvalues": holm_adjust(pvalues),
            "interaction_sequence_deltas": interaction,
            "interaction_mean": float(np.mean(list(interaction.values()))) if interaction else None,
            "interaction_bootstrap_95_ci": sequence_bootstrap_interval(interaction, seed=20261107) if interaction else None,
            "selector_discrimination_signal": bool(signal),
            "next": "bounded pulse diagnostic requires a separately frozen target plan; no automatic efficacy launch" if signal
                    else "finish planned diagnostic; if complete, stop selector development pending mechanism analysis"}


def report(out, result):
    lines = ["# Run171: memory-token group contribution", "",
             f"Sequences completed: {result['complete_sequences']}/21. Fixed native histories; no treatment writeback.", "",
             "| Removal | Present Dice | Boundary F | False alarms / empty targets |",
             "|---|---:|---:|---:|"]
    fmt = lambda x: "N/A" if x is None else f"{x:.6f}"
    for arm, row in result['means'].items():
        lines.append(f"| {arm} | {fmt(row['dice'])} | {fmt(row['boundary_f'])} | {row['false_alarms']}/{row['empty_frames']} |")
    lines += ["", "| Sequence | Native | Remove rejected | Remove retained | Remove random |", "|---|---:|---:|---:|---:|"]
    for name in result['per_sequence']['native']:
        lines.append(f"| {name} | " + " | ".join(fmt(result['per_sequence'][a][name]['dice']) for a in ('native','rejected','retained','random')) + " |")
    for label, c in result['primary_contrasts'].items():
        lines += ["", f"{label}: {c['mean']:+.6f}; sequence-bootstrap 95% CI {c['bootstrap_95_ci']}; {c['wins']}/{c['n']} sequence wins."]
    lines += ['', '| Removal | Hurts / helps / exact Dice ties (present targets) | Median native − removed Dice |',
              '|---|---:|---:|']
    for arm,u in result['target_utility_distribution'].items():
        lines.append(f"| {arm} | {u['removal_hurts']} / {u['removal_helps']} / {u['exact_dice_ties']} | {fmt(u['median_native_minus_removed_dice'])} |")
    lines += ['', 'Target counts are descriptive; frames are correlated and equal Dice need not mean identical masks.', '',
              '| Removed groups | Selector-retained fraction | First-read usage mass | Soft foreground | Vector norm | Frame age |',
              '|---|---:|---:|---:|---:|---:|']
    for arm,d in result['removed_group_descriptors'].items():
        lines.append(f"| {arm} | " + ' | '.join(fmt(d[k]) for k in ('retained_fraction','first_read_usage_mass','mean_soft_foreground','mean_vector_norm','age'))+' |')
    lines += ["", "Blocks are enriched in retained/rejected tokens; they are not pure selector categories.",
              "Random controls match removal count, block size and frame age, not global connected geometry.",
              "Ground truth only stratifies targets and scores outputs; no oracle token selection.",
              "A zero effect does not prove dispensability. All results are exploratory.", "",
              f"Joint-removal nonadditivity (Dice): {result['interaction_mean']}; positive means joint removal hurts more than the sum of separate removals.",
              "", f"Next: {result['next']}.", "", f"Artifacts: `{out}`."]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')


def guard(core):
    for rel, sha in core['sources'].items():
        if digest(ROOT/rel) != sha:
            raise RuntimeError(f"Source changed during run: {rel}")


def status(out, phase, **kwargs):
    base.atomic_json(out/'run_status.json', {"phase":phase, "pid":os.getpid(),
        "updated_at":datetime.now(timezone.utc).isoformat(), **kwargs})


@torch.inference_mode()
def sequence(args, videos, name, repo, out, core, gate):
    frames, plan = videos[name], core['target_plan'][name]
    target = out/'sequences'/name
    target.mkdir(parents=True, exist_ok=False)
    references = {}
    if gate:
        model = base.init_predictor(args.checkpoint, repo)
        with tempfile.TemporaryDirectory(dir=target, prefix='native_reference_') as temp:
            state, prompt = base.prepare(model, frames, Path(temp))
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError('Object identity changed')
                references[frame] = tensor_hash(logits)
        del model, state
        gc.collect()
    model = base.init_predictor(args.checkpoint, repo)
    weights = parameter_digest(model)
    controller = TokenContributionRead(model)
    controller.sequence = name
    rows = {a:[] for a in ARMS}
    with tempfile.TemporaryDirectory(dir=target, prefix='native_frames_') as temp:
        state, prompt = base.prepare(model, frames, Path(temp))
        if prompt != plan['prompt']:
            raise RuntimeError('Prompt changed')
        iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
        expected = prompt
        while True:
            controller.arm = 'collect'
            with controller:
                try:
                    frame, ids, logits = next(iterator)
                except StopIteration:
                    break
            if ids != [1] or frame != expected:
                raise RuntimeError('Native trajectory identity/order changed')
            expected += 1
            if gate and tensor_hash(logits) != references[frame]:
                raise RuntimeError(f'Read-only collector changed native prediction {name}/{frame}')
            if frame == prompt:
                continue
            collect_trace = controller.calls[-1]
            selected = frame in plan['targets']
            if selected or gate:
                sample = capture_training_read(model, state, frame)
                signature = lambda: fingerprint(bank_contents(state,frame,[],model.max_obj_ptrs_in_encoder))['sha256']
                before = signature()
                controller.arm = 'native'
                native, trace = replay(model,sample,controller)
                check_exact(native,state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame])
                check_layout(trace,collect_trace)
                if tensor_hash(model._get_orig_video_res_output(state,native['pred_masks'])[1]) != tensor_hash(logits):
                    raise RuntimeError('Native full-resolution replay mismatch')
                if selected:
                    for arm in ARMS:
                        controller.arm = arm
                        result, arm_trace = (native,trace) if arm == 'native' else replay(model,sample,controller)
                        check_layout(arm_trace,trace)
                        if arm == 'random' and frame == plan['targets'][0]:
                            repeated,repeated_trace = replay(model,sample,controller)
                            check_exact(result,repeated)
                            if arm_trace != repeated_trace:
                                raise RuntimeError('Nondeterministic intervention')
                        full = model._get_orig_video_res_output(state,result['pred_masks'])[1]
                        row = scored(full,frames,frame,target/arm/'masks',arm_trace)
                        row['object_score_logit'] = float(result['object_score_logits'].item())
                        rows[arm].append(row)
                        with (target/arm/'frames.jsonl').open('a') as stream:
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                    for layer in range(4):
                        plans = {a:rows[a][-1]['read']['selection_layers'][layer] for a in ARMS}
                        if not (plans['rejected']['counts'] == plans['retained']['counts'] == plans['random']['counts']):
                            raise RuntimeError('Unmatched key budgets')
                        if sum(plans['native']['counts']) <= sum(plans['rejected']['counts']):
                            raise RuntimeError('Diagnostic removal was inactive')
                if signature() != before:
                    raise RuntimeError('Treatment changed native bank/cache')
            controller.expire(frame)
            controller.calls.clear()
            status(out,'engineering' if gate else 'cohort',sequence=name,frame=frame,
                   total_frames=len(frames),completed_targets=len(rows['native']),planned_targets=len(plan['targets']))
            if frame % 10 == 0 or selected:
                print(f"[{name}] native {frame}/{len(frames)-1}; targets {len(rows['native'])}/{len(plan['targets'])}",flush=True)
        if expected != len(frames) or [r['frame'] for r in rows['native']] != plan['targets']:
            raise RuntimeError('Incomplete native trajectory or sampled targets')
    if parameter_digest(model) != weights:
        raise RuntimeError('Frozen weights changed')
    if base.input_hashes(videos,[name])[name] != core['inputs'][name]:
        raise RuntimeError('Input files changed')
    receipt = {"complete":True,"engineering_reference_checked":gate,"native_frames":len(frames)-prompt,
               "targets":len(rows['native']),"parameter_sha256":weights,
               "frames_sha256":{a:digest(target/a/'frames.jsonl') for a in ARMS}}
    base.atomic_json(target/'completion.json',receipt)
    del model,state,controller
    gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    os.environ['TMPDIR'] = str(out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    completed = {}
    try:
        status(out,'preflight')
        for rel in SOURCES:
            committed = subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT)
            if committed != (ROOT/rel).read_bytes():
                raise RuntimeError(f'Commit unchanged protocol/code first: {rel}')
        protocol = json.loads((ROOT/PROTOCOL).read_text())
        if digest(args.checkpoint) != protocol['checkpoint_sha256'] or digest(ROOT/'data/polypgen_sequence.jsonl') != protocol['manifest_sha256']:
            raise RuntimeError('Checkpoint/manifest does not match frozen protocol')
        videos = load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev') | load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')
        planned = target_plan(videos)
        if planned != protocol['target_plan'] or list(ARMS) != protocol['arms']:
            raise RuntimeError('Targets or arms differ from committed protocol')
        paths = sorted(set(base.SOURCE_REL)|set(EXTRA_SOURCES)|set(SOURCES))
        core = {"protocol":protocol,"checkpoint":str(args.checkpoint),"target_plan":planned,
                "inputs":base.input_hashes(videos,SEQUENCES),"sources":{p:digest(ROOT/p) for p in paths},
                "commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                "torch":torch.__version__,"tracking":"Drive JSON status/receipts; no network-dependent logger"}
        base.atomic_json(out/'setup.json',core)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for rel in paths:
                archive.write(ROOT/rel,arcname=rel)
        repo,_ = base.activate_author_repo()
        for name in SEQUENCES:
            if name not in ('seq2','seq3') and not (out/'engineering_pass.json').exists():
                raise RuntimeError('Cohort launch refused without engineering pass')
            guard(core)
            completed[name] = sequence(args,videos,name,repo,out,core,name in ('seq2','seq3'))
            result = summarize(completed)
            base.atomic_json(out/'partial_results.json',result)
            report(out,result)
            if name == 'seq3':
                base.atomic_json(out/'engineering_pass.json',{"passed":True,"sequences":['seq2','seq3'],
                    "setup_sha256":digest(out/'setup.json'),"no_outcome_based_gate":True})
        guard(core)
        result = summarize(completed)
        base.atomic_json(out/'results.json',result)
        report(out,result)
        base.atomic_json(out/'completion.json',{"complete":True,"elapsed_seconds":time.monotonic()-started,
            "results_sha256":digest(out/'results.json'),"setup_sha256":digest(out/'setup.json')})
        status(out,'complete',sequences=21)
    except BaseException as exc:
        base.atomic_json(out/'failure.json',{"error":repr(exc),"no_bypass":True})
        status(out,'failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
