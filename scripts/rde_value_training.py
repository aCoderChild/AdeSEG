"""Autoregressive training glue reusing native SAM2, its trainer and RDE loss.

One predicted frame/write per step (TBPTT1). GT values are detached loss targets
in a separate cache, never predictor memory. All teacher encodings occur after
the student forward and outside the native-input capture context.
"""
import torch

from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.complementary_memory.training import reader_clip_loss,detach_tensors,clone_tensors
from modeling.mavos_focal_value_modulation import MAVOSMemoryValueModulation
from modeling.native_memory_layout import selected_memory_frames
from modeling.rde_value_repair import bind_native_reads,value_consistency
from scripts.mask_memory_teacher import capture_native_mask_inputs,same_frame_teachers
from scripts.memory_value_query import spatial_values
from scripts.run_token_conditioned_key_adapter import _tree_digest


def aligned_teacher(bank,spans,cache,target_frame,object_id):
    """Fail on a wrong source frame, object bank, or replaced native write."""
    teachers={}
    for span in spans:
        f=span['frame']
        if f>=target_frame:
            raise RuntimeError('Future/current source in training teacher')
        if f in bank['cond_frame_outputs']:
            continue
        entry=cache.get(f)
        if entry is None or entry['frame']!=f or entry['object_id']!=object_id:
            raise RuntimeError('Missing or misassociated source teacher')
        native=bank['non_cond_frame_outputs'][f]
        if entry['native_memory_sha256']!=_tree_digest(native['maskmem_features']) or entry['native_score_sha256']!=_tree_digest(native['object_score_logits']):
            raise RuntimeError('Teacher belongs to another native source/object write')
        if entry['value'].requires_grad:
            raise RuntimeError('Teacher must be detached')
        teachers[f]={'mask_only':entry['value']}
    return spatial_values(bank,spans,teachers,'mask_only')


def training_step(model,adapter,optimizer,sample,bank,teacher_cache,criterion,consistency_weight):
    """Predict/write from own masks, then supervise; return detached resume state."""
    if sample['frame'] in bank['non_cond_frame_outputs']:
        raise RuntimeError('Current target is already stored in the bank')
    if sample.get('mask_inputs') is not None or sample.get('point_inputs') is not None:
        raise RuntimeError('Postprompt GT cannot enter the student predictor')
    if not 0<=consistency_weight:
        raise ValueError('Nonnegative consistency weight required')
    frame=sample['frame'];spans=selected_memory_frames(model,frame,bank,sample['num_frames'])
    repair_frames=[s['frame'] for s in spans if s['frame'] not in bank['cond_frame_outputs']]
    bank_before=_tree_digest(bank)
    observer=NativeComplementaryRead(model,None,trace_only=True)
    hook=MAVOSMemoryValueModulation(model,[adapter]*4,record_value_receipts=True,capture_adapted_values=True)
    optimizer.zero_grad(set_to_none=True)
    with bind_native_reads(model,adapter),observer,hook:
        # This scope contains only the actual student prediction/write.
        with capture_native_mask_inputs(model) as capture:
            segmentation,new_bank,writes=reader_clip_loss(model,[sample],bank)
        if capture['count']!=1:
            raise RuntimeError('Expected one actual predicted-memory write')
        native_capture=clone_tensors(capture['latest'])
        written=new_bank['non_cond_frame_outputs'][frame]
        if native_capture['stored_memory_sha256']!=_tree_digest(written['maskmem_features']):
            raise RuntimeError('Actual student encoder input/output association differs')
        prediction_sha256=_tree_digest(written)
        if len(hook.adapted_values)!=4 or len(observer.calls)!=1 or observer.calls[0]['layers']!=[0,1,2,3]:
            raise RuntimeError('Training did not visit four native layers')
        if not observer.calls[0]['native_positions_verified'] or not observer.calls[0]['stored_mask_alignment']:
            raise RuntimeError('Training memory/position alignment failed')
        if len(hook.value_calls)!=4 or any(not x['pointer_tail_bit_identical'] for x in hook.value_calls):
            raise RuntimeError('Training changed pointer values')
        # The target is assembled only after the student forward. It is never
        # passed to track_step, the adapter, or its returned predicted bank.
        target=aligned_teacher(bank,spans,teacher_cache,frame,sample['object_id'])
        terms=[]
        for item in hook.adapted_values:
            student=item['adapted'][:target.shape[0]]
            for span in spans:
                if span['frame'] in bank['cond_frame_outputs'] and not torch.equal(student[span['start']:span['stop']],target[span['start']:span['stop']]):
                    raise RuntimeError('Conditioning values were repaired')
            if repair_frames:
                terms.append(value_consistency(student,target,adapter._repair_indices,criterion))
        consistency=torch.stack(terms).mean() if terms else segmentation.new_zeros(())
        loss=segmentation+consistency_weight*consistency
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
    # Generate this source's target only after its own prediction was committed.
    # same_frame_teachers calls the frozen author encoder and checks native
    # re-encoding exactly; it never inserts GT values into new_bank.
    visual={'vision':sample['vision'],'positions':sample['positions'],'sizes':sample['sizes']}
    teacher=same_frame_teachers(model,visual,written,sample['teacher_target'],native_capture)
    if _tree_digest(written)!=prediction_sha256 or _tree_digest(bank)!=bank_before:
        raise RuntimeError('Auxiliary teacher generation changed predicted memory')
    cache=dict(teacher_cache)
    cache[frame]={'frame':frame,'object_id':sample['object_id'],'native_memory_sha256':_tree_digest(written['maskmem_features']),
        'native_score_sha256':_tree_digest(written['object_score_logits']),
        'value':teacher['mask_only'].detach(),'gt_source_sha256':sample['teacher_target_sha256']}
    gradients={}
    if repair_frames:
        if not loss.requires_grad:raise RuntimeError('Repaired tokens have no training graph')
        loss.backward()
        gradients={n:float(p.grad.detach().abs().sum()) for n,p in adapter.named_parameters() if p.grad is not None}
        if any(not torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite adapter gradient')
        if adapter.mlp.layers[-1].weight.grad is None:raise RuntimeError('No MLP output gradient')
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),3)
        optimizer.step()
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('SAM2 received gradients')
    if frame+1<sample['num_frames']:
        next_frames={s['frame'] for s in selected_memory_frames(model,frame+1,new_bank,sample['num_frames'])}
        cache={f:entry for f,entry in cache.items() if f in next_frames}
    else:cache={}
    row={'frame':frame,'segmentation_loss':float(segmentation.detach()),'consistency_loss':float(consistency.detach()),
        'loss':float(loss.detach()),'consistency_weight':consistency_weight,'gradient_l1':gradients,
        'optimizer_updated':bool(repair_frames),'repair_frames':repair_frames,'memory_reads':observer.calls,
        'value_calls':hook.value_calls,'prediction_sha256':prediction_sha256,'writes':writes,
        'native_source_reencoding_exact':True,'student_encoder_calls_captured':capture['count'],
        'teacher_used_for_prediction_or_bank':False,'teacher_cache_frames':sorted(cache),
        'new_bank_sha256':_tree_digest(new_bank),'teacher_cache_sha256':_tree_digest(cache)}
    return detach_tensors(new_bank),cache,row
