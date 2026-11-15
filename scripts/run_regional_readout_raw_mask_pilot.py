"""Train and compare query-dependent regional memory readouts on PolypGen."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cfbi_training import object_presence_loss
from modeling.decoder_training_mask import decoder_training_mask, select_decoder_mask
from modeling.native_region_readout import NativeRegionalReadout, RegionalReadoutHeads
from modeling.regional_refinement import refinement_loss
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import load, save_tensor, write
import scripts.run_cfbi_region_pilot as pilot

SOURCES = ['protocols/regional_readout_raw_mask_pilot.json', 'scripts/run_regional_readout_raw_mask_pilot.py',
           'modeling/decoder_training_mask.py', 'tests/test_decoder_training_mask.py',
           'scripts/run_cfbi_region_pilot.py', 'modeling/native_region_readout.py',
           'modeling/region_readout.py', 'modeling/cfbi_region_gate.py', 'modeling/cfbi_training.py',
           'modeling/region_definitions.py', 'modeling/region_tokens.py', 'tests/test_region_readout.py',
           'tests/test_cfbi_region_gate.py', 'tests/test_cfbi_training.py']


class PilotReadout(NativeRegionalReadout):
    @property
    def trace(self):
        return self.observer.trace


def train(args, samples, schedule, repo, mode):
    model = build_author_predictor(args, repo)
    model.requires_grad_(False)
    heads = RegionalReadoutHeads().train()
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.001)
    losses, gradients = [], []
    for step, index in enumerate(schedule):
        row = samples[index]
        sample = load(row['path'], row['sha256'])
        bank = fingerprint(sample['output_dict'])['sha256']
        optimizer.zero_grad()
        with decoder_training_mask(model.sam_mask_decoder) as masks, NativeRegionalReadout(model, heads, mode, row['sequence']) as adapter:
            output = model.track_step(frame_idx=sample['frame'], is_init_cond_frame=False,
                current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                track_in_reverse=False, run_mem_encoder=False)
        if step == 0 and not torch.equal(output['pred_masks'].detach(), sample['native_logits']):
            raise RuntimeError('Zero-residual native replay mismatch')
        if [r['layer'] for r in adapter.calls] != [0, 1, 2, 3]:
            raise RuntimeError('Incomplete attention layer coverage')
        loss = refinement_loss(select_decoder_mask(masks), sample['target']) + object_presence_loss(output['object_score_logits'], sample['target'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norms = [float(x.mix.weight.grad.abs().sum()) if x.mix.weight.grad is not None else 0.0 for x in heads.layers]
        if not all(np.isfinite(x) for x in norms):
            raise RuntimeError('Nonfinite gradients')
        if any(p.grad is not None for p in model.parameters()) or fingerprint(sample['output_dict'])['sha256'] != bank:
            raise RuntimeError('Frozen backbone or memory changed')
        optimizer.step()
        losses.append(float(loss.detach()))
        gradients.append(norms)
        write(args.output_dir / 'progress.json', {'phase': 'training', 'mode': mode, 'step': step + 1, 'total': len(schedule)})
    if not all(any(row[i] > 0 for row in gradients) for i in range(4)):
        raise RuntimeError('At least one head received no gradient')
    folder = args.output_dir / mode
    folder.mkdir()
    save_tensor(heads.state_dict(), folder / 'head.pt')
    write(folder / 'training.json', {'schedule': schedule, 'losses': losses, 'gradient_l1': gradients,
                                     'checkpoint_sha256': digest(folder / 'head.pt')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior-dir', 'reference-setup', 'source', 'gates', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        args.method, args.device = 'sam2', 'cpu'
        for filename in ('readout_zero_full_seq2_gate_v1/completion.json', 'readout_nonzero_seq2_gate_v1/completion.json',
                         'native_region_readout_gradient_gate.json', 'native_region_readout_missed_gradient_gate.json',
                         'native_region_readout_shuffled_gate.json', 'native_region_readout_generic_gate.json'):
            if not json.loads((args.gates / filename).read_text())['passed']:
                raise RuntimeError(f'Gate failed: {filename}')
        zero_gate = json.loads((args.gates / 'readout_zero_full_seq2_gate_v1/completion.json').read_text())
        nonzero_gate = json.loads((args.gates / 'readout_nonzero_seq2_gate_v1/completion.json').read_text())
        if zero_gate['sequence'] != 'seq2' or zero_gate['trained_one_step'] or len(zero_gate['frames']) != 59 or not all(
            frame['native_identical'] for frame in zero_gate['frames']):
            raise RuntimeError('Full native identity gate is incomplete')
        if nonzero_gate['sequence'] != 'seq2' or not nonzero_gate['trained_one_step'] or not nonzero_gate['frames']:
            raise RuntimeError('Trained-residual propagation gate is incomplete')
        hashes = {path: digest(ROOT / path) for path in SOURCES}
        if any(digest(ROOT / path) != value for path, value in nonzero_gate['sources'].items()):
            raise RuntimeError('Nonzero gate sources changed')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_region_readout.py',
                        'tests/test_cfbi_region_gate.py', 'tests/test_cfbi_training.py', 'tests/test_decoder_training_mask.py'], cwd=ROOT, check=True)
        if hashes != {path: digest(ROOT / path) for path in hashes}:
            raise RuntimeError('Sources changed during tests')
        protocol = json.loads((ROOT / SOURCES[0]).read_text())
        previous = json.loads((args.prior_dir / 'setup.json').read_text())
        reference = json.loads(args.reference_setup.read_text())
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        if digest(args.checkpoint) != previous['checkpoint_sha256'] or digest(manifest) != previous['manifest_sha256']:
            raise RuntimeError('Checkpoint or manifest changed')
        samples, schedule = previous['samples'], previous['schedule']
        if len(samples) != 14 or len(schedule) != 24 or any(digest(r['path']) != r['sha256'] for r in samples):
            raise RuntimeError('Saved training samples or schedule invalid')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        inputs = args.source / 'inputs.json'
        if digest(inputs) != reference['inputs_sha256']:
            raise RuntimeError('Input receipts changed')
        receipts = json.loads(inputs.read_text())
        for name in protocol['training_sequences'] + protocol['validation_sequences']:
            for frame, receipt in zip(videos[name], receipts[name], strict=True):
                if frame.frame_index != receipt['frame'] or digest(frame.image_path) != receipt['image_sha256'] or digest(frame.mask_path) != receipt['mask_sha256']:
                    raise RuntimeError(f'Input changed: {name}/{frame.frame_index}')
        native = {}
        for name in protocol['validation_sequences']:
            path = args.source / 'cache' / name / 'native_frames.json'
            if digest(path) != reference['validation_native_frames_sha256'][name]:
                raise RuntimeError(f'Native baseline changed: {name}')
            rows = json.loads(path.read_text())
            pilot.verify_baseline_coverage(rows, prompt_index(videos[name], {'initialization': 'first_labeled_frame'}), len(videos[name]))
            native[name] = {**propagation_summary(rows), **{k: float(np.mean([r['scores'][k] for r in rows if r['present']])) for k in ('iou', 'boundary_f')}}
        repo, revision = pilot.activate_author_repo()
        write(args.output_dir / 'setup.json', {'protocol': protocol, 'sources': hashes, 'checkpoint_sha256': previous['checkpoint_sha256'],
              'manifest_sha256': previous['manifest_sha256'], 'sam2_revision': revision, 'samples': samples, 'schedule': schedule,
              'patient_independence_verified': False})
        for mode in ('regional', 'shuffled', 'generic'):
            torch.manual_seed(20261031)
            train(args, samples, schedule, repo, mode)
        pilot.RegionGate, pilot.NativeCFBIGate = RegionalReadoutHeads, PilotReadout
        results = {'native': native, 'conditions': {}, 'scope': protocol['limitations']}
        for mode in ('regional', 'shuffled', 'generic'):
            info = json.loads((args.output_dir / mode / 'training.json').read_text())
            results['conditions'][mode] = {}
            for name in protocol['validation_sequences']:
                rows, summary = pilot.validate(args, videos[name], repo, mode, name, args.output_dir / mode / 'head.pt', info['checkpoint_sha256'])
                write(args.output_dir / mode / f'{name}_frames.json', rows)
                results['conditions'][mode][name] = summary
        if hashes != {path: digest(ROOT / path) for path in hashes} or digest(inputs) != reference['inputs_sha256'] or any(
            digest(args.source / 'cache' / name / 'native_frames.json') != reference['validation_native_frames_sha256'][name]
            for name in protocol['validation_sequences']):
            raise RuntimeError('Sources changed during pilot')
        write(args.output_dir / 'results.json', results)
        write(args.output_dir / 'completion.json', {'complete': True, 'validation_sequences': protocol['validation_sequences']})
        write(args.output_dir / 'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
