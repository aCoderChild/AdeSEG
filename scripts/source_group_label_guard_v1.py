"""CPython open-audit guard against reading PolypGen manifest masks at inference.

The guard covers Python `open` audit events, not arbitrary native-code system
calls; it is an engineering control alongside source review, not a proof that
all third-party native code avoids labels.
"""
import json
import os
from pathlib import Path
import sys


def install_mask_open_guard(manifest):
    manifest=Path(manifest);blocked=set()
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);path=Path(row['mask'])
        if not path.is_absolute():path=manifest.parent/path
        blocked.add(os.path.realpath(path))
    counts={'blocked_paths':len(blocked),'open_attempts':0}
    def audit(event,args):
        if event!='open' or not args:return
        path=args[0]
        if not isinstance(path,(str,bytes,os.PathLike)):return
        if os.path.realpath(os.fsdecode(path)) in blocked:
            counts['open_attempts']+=1
            raise PermissionError('Ground-truth mask open blocked during capture')
    sys.addaudithook(audit)
    return counts
