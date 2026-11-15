#!/usr/bin/env python3
"""Wrap the existing warm-capture runner with a separately proven bounded loader.

This draft is not authorized for standalone execution. Its enclosing,
source-bound batch launcher and independent full-sequence auditor must be
completed and reviewed before use. It never edits the frozen V2 capture code.
"""
import argparse
import fcntl
import os
import psutil
import hashlib
import json
import resource
from pathlib import Path
import subprocess
import sys
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from modeling.bounded_sam2_video_loader_v2 import install_bounded_eager_loader, resolve_swr_init_chain
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, validate_plan
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_crossfit_warm_video_v2 import FORK, video_inputs, warm_plan
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_bounded_loader_gate_v2 import verify_gate
from scripts.warm_crossfit_video_contract_v1 import require
import scripts.run_cache_release_warm_base_v1 as original

PROTOCOL = 'protocols/CACHE_RELEASE_FULL_SEQ6_V1.md'
ENABLE_REVIEWED_EXECUTION = True  # Independently reviewed for supervised seq6 only.
ENTRYPOINTS = ('scripts/verify_mps_memory_diagnostic_v3.py','scripts/run_cache_release_full_v1.py', 'scripts/verify_bounded_loader_gate_v2.py',
               'modeling/bounded_sam2_video_loader_v2.py', 'scripts/audit_cache_release_full_v1.py',
               'scripts/run_cache_release_full_study_v1.py', 'scripts/run_cache_release_warm_base_v1.py',
               'scripts/audit_cache_release_warm_base_v1.py')


def sources():
    paths = set(local_import_closure(ROOT, ENTRYPOINTS)); paths.add(PROTOCOL)
    return {p: digest(ROOT/p) for p in sorted(paths)}




def assert_live_supervisor(manifest):
    """Require the frozen direct parent and an actively held shared experiment lock."""
    pid=manifest.get('supervisor_pid'); created=manifest.get('supervisor_create_time')
    require(type(pid) is int and pid>0 and isinstance(created,(int,float)) and pid==os.getppid(),
            'Cache-release supervisor parent differs')
    process=psutil.Process(pid); command=process.cmdline()
    require(process.is_running() and process.create_time()==created and len(command)>=2 and
            (Path(process.cwd())/command[1]).resolve()==ROOT/'scripts/run_cache_release_full_study_v1.py',
            'Cache-release supervisor identity differs')
    lock_path=OUT/'.source_group_capture_resume_v1.lock'
    with lock_path.open('r') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
            raise ValueError('Cache-release supervisor lock is not held')


def validate_supervision(*, sequence, mode, output, certificate_path, short_gate_path, reference, cache_frames):
    """Fail closed for every callable/CLI route outside its frozen supervised phase."""
    require(ENABLE_REVIEWED_EXECUTION, 'Cache-release full execution is disabled pending review')
    out=Path(output).resolve(); root=out.parent.parent.resolve(strict=True)
    require(sequence=='seq6' and mode in ('reference','capture') and out==root/sequence/mode and
            root.is_relative_to(OUT.resolve(strict=True)), 'Invalid supervised cache-release output')
    manifest_path=root/'run_manifest.json'; sha=digest(manifest_path); manifest=json.loads(manifest_path.read_text())
    require(digest(manifest_path)==sha and manifest.get('sequence')==sequence and
            manifest.get('certificate_path')==str(Path(certificate_path).resolve(strict=True)) and
            manifest.get('short_gate_path')==str(Path(short_gate_path).resolve(strict=True)) and
            manifest.get('cache_frames')==cache_frames and manifest.get('source_sha256')==sources(),
            'Cache-release supervised manifest differs')
    assert_live_supervisor(manifest)
    _,cert_sha=verify_certificate(certificate_path); _,gate_sha=verify_gate(short_gate_path)
    from scripts.verify_mps_memory_diagnostic_v3 import verify_gate as verify_memory_gate
    memory,memory_sha=verify_memory_gate(manifest['memory_gate_path'])
    require(manifest.get('certificate_sha256')==cert_sha and manifest.get('short_gate_sha256')==gate_sha and
            manifest.get('memory_gate_sha256')==memory_sha and memory.get('certificate_sha256')==cert_sha,
            'Cache-release supervised proof differs')
    ledger=root/'phases.jsonl'; rows=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    expected=[] if mode=='reference' else ['seq6_reference']
    require([x.get('phase') for x in rows]==expected, 'Cache-release supervised phase differs')
    if mode=='reference': require(reference is None, 'Reference cannot have prior reference')
    else:
        from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
        previous=root/sequence/'reference'
        require(reference is not None and Path(reference).resolve(strict=True)==previous and
                phase_artifact(root,expected[0],previous/'receipt.json')==rows[0]['artifact_sha256'] and
                digest(previous/'bounded_receipt.json')==rows[0]['bounded_receipt_sha256'],
                'Cache-release reference is not phase-bound')
        side_path=previous/'bounded_receipt.json'; side=json.loads(side_path.read_text())
        require(digest(side_path)==rows[0]['bounded_receipt_sha256'] and side.get('passed') is True and
                side.get('source_sha256')==manifest['source_sha256'] and side.get('certificate_sha256')==cert_sha and
                side.get('short_gate_sha256')==gate_sha and side.get('label_guard',{}).get('open_attempts')==0,
                'Cache-release reference sidecar differs')
    return manifest,sha


def run_capture(*, sequence, mode, output, certificate_path, short_gate_path,
                reference=None, cache_frames=4):
    """Run the original capture code with a scoped, provenance-bound loader."""
    require(sequence == 'seq6' and mode in ('reference','capture') and
            (mode == 'capture') == (reference is not None) and 1 <= cache_frames <= 4,
            'Invalid bounded capture configuration')
    validate_supervision(sequence=sequence,mode=mode,output=output,certificate_path=certificate_path,
                         short_gate_path=short_gate_path,reference=reference,cache_frames=cache_frames)
    out = Path(output).resolve(); require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(),
                                          'Invalid bounded capture output')
    guard = install_mask_open_guard(DATA)
    warm = warm_plan(); index = validate_plan(); certificate, certificate_sha = verify_certificate(certificate_path)
    gate, gate_sha = verify_gate(short_gate_path)
    gate_root = Path(short_gate_path).resolve(strict=True).parent.parent
    gate_manifest = json.loads((gate_root/'run_manifest.json').read_text())
    loader_file = ROOT/'modeling/bounded_sam2_video_loader_v2.py'
    require(gate.get('certificate_sha256') == certificate_sha and
            gate_manifest.get('source_sha256', {}).get('modeling/bounded_sam2_video_loader_v2.py') == digest(loader_file),
            'Short loader gate does not bind this certificate and loader')
    frames, records, _, _, _, _, _, _ = video_inputs(sequence, index, warm)
    source_map = sources(); holder = {}; original_builder = original.build_author_predictor
    predictor_file = FORK/'sam2/sam2_video_predictor_topk_insurance.py'
    predictor_sha = warm['old_identity']['fork_patches']['sam2/sam2_video_predictor_topk_insurance.py']
    wrapper_file = FORK/'sam2/sam2_video_predictor_swr_wrapper.py'; wrapper_sha = digest(wrapper_file)
    misc_file = FORK/'sam2/utils/misc.py'; misc_sha = digest(misc_file)
    for relative, expected in (('sam2/sam2_video_predictor_swr_wrapper.py', wrapper_sha),
                               ('sam2/utils/misc.py', misc_sha)):
        tracked = subprocess.check_output(['git','-C',str(FORK),'show','HEAD:'+relative])
        require(hashlib.sha256(tracked).hexdigest() == expected, 'Pinned author loader source differs')
    def bounded_builder(arguments, repo):
        require('model' not in holder, 'Unexpected repeated predictor construction')
        model = original_builder(arguments, repo); holder['model'] = model
        resolve_swr_init_chain(type(model), wrapper_file=wrapper_file, wrapper_sha256=wrapper_sha,
                               predictor_file=predictor_file, predictor_sha256=predictor_sha)
        from sam2.utils import misc
        require(Path(misc.__file__).resolve() == misc_file.resolve(), 'Wrong SAM2 decoder')
        original_init = model.init_state
        def bounded_init(self, *positional, **kwargs):
            require('loader' not in holder, 'Unexpected repeated predictor initialization')
            staged = kwargs.get('video_path', positional[0] if positional else None)
            require(staged is not None, 'Missing staged video path')
            frame_bytes = 3*self.image_size*self.image_size*4
            with install_bounded_eager_loader(self, staged_dir=staged, expected_paths=[f.image_path for f in frames],
                    expected_sha256=[r['image_sha256'] for r in records], author_module_file=predictor_file,
                    author_module_sha256=predictor_sha, author_misc_sha256=misc_sha,
                    wrapper_module_file=wrapper_file, wrapper_module_sha256=wrapper_sha,
                    author_load_video_frames=misc.load_video_frames, author_decoder=misc._load_img_as_tensor,
                    limit_bytes=cache_frames*frame_bytes) as created:
                state = original_init(*positional, **kwargs)
            require(state['images'] is created[0], 'Bounded state was not installed')
            holder['loader'] = created[0]
            return state
        model.init_state = MethodType(bounded_init, model)
        original_propagate = model.propagate_in_video
        holder['sampled_peak_mps_allocated_bytes'] = 0; holder['sampled_peak_mps_driver_bytes'] = 0
        def monitored_propagate(self, *positional, **kwargs):
            for item in original_propagate(*positional, **kwargs):
                holder['sampled_peak_mps_allocated_bytes'] = max(holder['sampled_peak_mps_allocated_bytes'],
                    int(torch.mps.current_allocated_memory()))
                holder['sampled_peak_mps_driver_bytes'] = max(holder['sampled_peak_mps_driver_bytes'],
                    int(torch.mps.driver_allocated_memory()))
                yield item
        model.propagate_in_video = MethodType(monitored_propagate, model)
        return model
    arguments = ['run_cache_release_warm_base_v1.py', '--sequence', sequence, '--mode', mode,
                 '--certificate', str(Path(certificate_path).resolve()), '--output', str(out),
                 '--batch-root',str(out.parent.parent),'--short-gate',str(Path(short_gate_path).resolve()),
                 '--cache-frames',str(cache_frames)]
    if reference is not None: arguments += ['--reference', str(Path(reference).resolve())]
    old_argv = sys.argv; original.build_author_predictor = bounded_builder
    try:
        sys.argv = arguments
        original.main()
    finally:
        original.build_author_predictor = original_builder; sys.argv = old_argv
    require('loader' in holder and (out/'receipt.json').is_file(), 'Bounded original capture did not complete')
    holder['loader'].verify_sources(); cache = holder['loader'].stats()
    original_receipt_path=out/'receipt.json';original_receipt_sha=digest(original_receipt_path)
    original_receipt=json.loads(original_receipt_path.read_text())
    require(digest(original_receipt_path)==original_receipt_sha and
            original_receipt.get('process_id')==__import__('os').getpid() and
            isinstance(original_receipt.get('run_uuid'),str) and bool(original_receipt['run_uuid']) and
            original_receipt.get('bounded_base_source_sha256')==
                digest(ROOT/'scripts/run_cache_release_warm_base_v1.py'),
            'Bounded base receipt process, UUID or source differs')
    require(cache['final_sha256_verified'] is True and cache['evictions'] > 0 and guard['open_attempts'] == 0 and sources() == source_map and
            digest(wrapper_file) == wrapper_sha and digest(misc_file) == misc_sha,
            'Bounded capture source, cache or final verification differs')
    receipt = {'passed': True, 'scope': 'Bounded-loader wrapper around original warm capture; not yet full-sequence compatibility proof.',
        'sequence': sequence, 'mode': mode, 'original_receipt_sha256': original_receipt_sha,
        'certificate_sha256': certificate_sha, 'short_gate_path': str(Path(short_gate_path).resolve()),
        'short_gate_sha256': gate_sha, 'loader_sha256': digest(loader_file), 'predictor_sha256': predictor_sha,
        'wrapper_sha256': wrapper_sha, 'misc_sha256': misc_sha, 'cache': cache, 'source_sha256': source_map,
        'label_guard': dict(guard),
        'process_id': original_receipt['process_id'], 'run_uuid': original_receipt['run_uuid'],
        'peak_process_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
        'sampled_peak_mps_allocated_bytes': holder['sampled_peak_mps_allocated_bytes'],
        'sampled_peak_mps_driver_bytes': holder['sampled_peak_mps_driver_bytes']}
    (out/'bounded_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', choices=('seq6','seq11'), required=True)
    parser.add_argument('--mode', choices=('reference','capture'), required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--short-gate', type=Path, required=True); parser.add_argument('--reference', type=Path)
    parser.add_argument('--batch-root', type=Path, required=True); parser.add_argument('--cache-frames', type=int, default=4)
    args = parser.parse_args(); outer_guard=install_mask_open_guard(DATA)
    require(args.sequence=='seq6','Only reviewed seq6 execution is enabled')
    root=args.batch_root.resolve(strict=True); out=args.output.resolve()
    require(root.is_relative_to(OUT.resolve(strict=True)) and out==root/args.sequence/args.mode,
            'Bounded capture output differs from supervised phase')
    manifest_path=root/'run_manifest.json';manifest_sha=digest(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    require(digest(manifest_path)==manifest_sha and manifest.get('sequence')==args.sequence and
            manifest.get('certificate_path')==str(args.certificate.resolve(strict=True)) and
            manifest.get('short_gate_path')==str(args.short_gate.resolve(strict=True)) and
            manifest.get('cache_frames')==args.cache_frames and manifest.get('source_sha256')==sources(),
            'Bounded capture batch manifest differs')
    _,certificate_sha=verify_certificate(args.certificate);_,gate_sha=verify_gate(args.short_gate)
    from scripts.verify_mps_memory_diagnostic_v3 import verify_gate as verify_memory_gate
    memory_gate,memory_sha=verify_memory_gate(manifest['memory_gate_path'])
    require(memory_sha==manifest.get('memory_gate_sha256') and memory_gate.get('certificate_sha256')==certificate_sha,
            'MPS memory diagnostic proof differs')
    require(manifest.get('certificate_sha256')==certificate_sha and manifest.get('short_gate_sha256')==gate_sha and
            outer_guard['open_attempts']==0,'Bounded capture certificate, short gate or label guard differs')
    ledger=root/'phases.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    expected=[] if args.mode=='reference' else [f'{args.sequence}_reference']
    require([x.get('phase') for x in rows]==expected, 'Bounded capture phase is not next')
    if args.mode=='capture':
        from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
        expected_reference=root/args.sequence/'reference'
        require(args.reference is not None and args.reference.resolve(strict=True)==expected_reference and
                phase_artifact(root,expected[0],expected_reference/'receipt.json')==rows[0]['artifact_sha256'] and
                digest(expected_reference/'bounded_receipt.json')==rows[0]['bounded_receipt_sha256'],
                'Bounded reference is not phase-bound')
        side=json.loads((expected_reference/'bounded_receipt.json').read_text())
        require(digest(expected_reference/'bounded_receipt.json')==rows[0]['bounded_receipt_sha256'] and
                side.get('passed') is True and side.get('sequence')==args.sequence and side.get('mode')=='reference' and
                side.get('certificate_sha256')==certificate_sha and side.get('short_gate_sha256')==gate_sha and
                side.get('source_sha256')==manifest['source_sha256'] and
                side.get('label_guard',{}).get('open_attempts')==0, 'Bounded reference sidecar differs')
    else:
        require(args.reference is None, 'Reference phase cannot accept a prior reference')
    if args.sequence=='seq11':
        from scripts.run_cache_release_full_study_v1 import verify_seq6_proof
        _,seq6_sha=verify_seq6_proof(Path(manifest['seq6_proof_path']),certificate_sha,gate_sha)
        require(seq6_sha==manifest.get('seq6_proof_sha256'),'Seq6 compatibility proof differs')
    require(ENABLE_REVIEWED_EXECUTION, 'Bounded full capture is not yet enabled pending independent review')
    run_capture(sequence=args.sequence,mode=args.mode,output=out,certificate_path=args.certificate,
                short_gate_path=args.short_gate,reference=args.reference,cache_frames=args.cache_frames)
    require(outer_guard['open_attempts']==0 and digest(manifest_path)==manifest_sha,
            'Bounded capture manifest or outer label guard changed')


if __name__ == '__main__': main()
