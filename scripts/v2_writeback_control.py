"""Oracle diagnostic glue for the upstream predictor's stored-frame dictionary."""
import copy
import json
from pathlib import Path
import zipfile

import torch
from scripts.bank_integrity import fingerprint

CORE = ('prediction_sha256', 'low_res_sha256', 'bank_sha256', 'cache_sha256')


class FrameWritebackControl:
    def __init__(self, destination, pulse_reference=None):
        self.destination = Path(destination)
        self.pulse_reference = pulse_reference
        self.snapshot = None

    def __call__(self, state, index, core, pulse):
        if index != 50:
            return None
        outputs = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
        current = outputs[index]
        required = {'pred_masks', 'maskmem_features', 'maskmem_pos_enc', 'obj_ptr', 'object_score_logits'}
        if not required.issubset(current) or pulse is None:
            raise RuntimeError('Missing complete frame50 output or pulse receipt')
        if self.pulse_reference is None:
            self.snapshot = {'output': copy.deepcopy(current), 'core': dict(core)}
            torch.save(self.snapshot, self.destination)
            return {'action': 'capture_native', 'stored_keys': sorted(current)}
        if self.snapshot is None:
            raise RuntimeError('Native frame50 capture required first')
        reference = self.pulse_reference
        if any(core[k] != reference[k] for k in CORE) or pulse != reference['pulse']:
            raise RuntimeError('Frame50 pre-replacement pulse differs from completed experiment')
        if set(current) != set(self.snapshot['output']):
            raise RuntimeError('Stored frame fields changed')
        # Replace the whole output, not a mask-only edit. Keep yielded logits untouched.
        outputs[index] = copy.deepcopy(self.snapshot['output'])
        after_bank = fingerprint(state['output_dict_per_obj'])['sha256']
        after_cache = fingerprint(state['cached_features'])['sha256']
        if after_bank != self.snapshot['core']['bank_sha256'] or after_cache != self.snapshot['core']['cache_sha256']:
            raise RuntimeError('Restored bank/cache is not exactly native at50')
        return {'action':'replace_whole_frame', 'stored_keys':sorted(current),
                'before_bank_sha256':core['bank_sha256'], 'after_bank_sha256':after_bank,
                'after_cache_sha256':after_cache, 'emitted_prediction_unchanged':True}


def run_control(args, output, execute):
    previous = args.writeback_control
    if not json.loads((previous/'validation.json').read_text())['passed']:
        raise ValueError('Completed transition validation required')
    with zipfile.ZipFile(previous/'transition_pulse_regional_first.zip') as z:
        reference = next(r for r in map(json.loads,z.read('integrity.jsonl').decode().splitlines()) if r['frame']==50)
    args.schedule = 'transition_pulse'
    capture = FrameWritebackControl(output/'native_frame50.pt')
    args.frame_callback = capture
    native, native_receipts = execute('seq13','sham',None,output/'gate/native')
    control = FrameWritebackControl(output/'native_frame50.pt',reference)
    control.snapshot = capture.snapshot
    args.frame_callback = control
    restored, first = execute('seq13','regional',None,output/'replacement/first')
    execute('seq13','regional',None,output/'replacement/repeat',repeat=first)
    # Equality after50 is an outcome, not an acceptance criterion.
    future = {k:all(first[f][k]==native_receipts[f][k] for f in first if f>50) for k in CORE}
    validation={'passed':True,'native_sham_exact':True,'pre_replacement_pulse_exact':True,
                'restored_frame50_bank_cache_exact':True,'fresh_repeat_exact':True,
                'future_native_equality':future}
    results={'completed':True,'exploratory':True,'native':native,'replacement':restored,
             'future_native_equality':future,'limitation':'Oracle whole-frame restoration; does not isolate vectors versus pointers or establish a deployable method.'}
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Frame50 whole-writeback replacement','',results['limitation'],'',
           '| Condition | Present Dice | Boundary F | False alarms |','|---|---:|---:|---:|']
    for label,s in [('Native sham',native),('Pulse + native writeback replacement',restored)]:
        w=s['whole'];lines.append(f"| {label} | {w['present_dice']:.6f} | {w['present_boundary_f']:.6f} | {w['gt_empty_false_alarm_count']} |")
    lines+=['',f'Exact native equality on frames51–249: {future}',
            'Frame50 metrics retain the pruned prediction, despite native output being stored for subsequent reads.']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
