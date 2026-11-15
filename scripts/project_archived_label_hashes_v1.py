"""Project label hashes from frozen frame metadata, without parsing outcomes.

This reads metadata bytes only, not annotation masks. The caller must first
enforce its fold's train/validation/test authorization and may request only
ordinals belonging to the currently authorized phase.
"""
import re
from pathlib import Path

from scripts.memory_token_io import digest


def project_label_hashes(binding,expected_count,permitted_ordinals):
    path=Path(binding['path']).resolve(strict=True);wanted=binding['sha256']
    if digest(path)!=wanted:raise ValueError('Historical frame metadata changed')
    if type(expected_count) is not int or expected_count<1:raise ValueError('Invalid frame count')
    ordinals=list(permitted_ordinals)
    if (len(ordinals)!=len(set(ordinals)) or any(type(i) is not int or not 0<=i<expected_count for i in ordinals)):
        raise ValueError('Invalid permitted frame ordinals')
    selected=set(ordinals);results={};count=0
    pattern=re.compile(rb'^\s*"label_sha256"\s*:\s*"([a-f0-9]{64})"\s*,?\s*$')
    with path.open('rb') as stream:
        for line in stream:
            if b'"label_sha256"' not in line:continue
            match=pattern.fullmatch(line.rstrip(b'\r\n'))
            if not match:raise ValueError('Unexpected historical label-hash syntax')
            if count in selected:results[count]=match.group(1).decode('ascii')
            count+=1
    if digest(path)!=wanted or count!=expected_count or set(results)!=selected:
        raise ValueError('Historical label-hash coverage changed')
    return results
