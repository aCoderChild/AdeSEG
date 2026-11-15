"""Wait for a specific cohort process, then run two bounded gate replays."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.run_regional_refinement import write

SEQUENCES = {f"seq{i}" for i in range(2,24) if i != 7}
ARMS = {"uniform_generic","uniform_regional","recent_dense_generic","recent_dense_regional"}


def verify_cohort_completion(folder):
    folder = Path(folder)
    done = json.loads((folder/"completion.json").read_text())
    progress = json.loads((folder/"progress.json").read_text())
    result = json.loads((folder/"results.json").read_text())
    if done.get("complete") is not True or progress.get("status") != "complete":
        raise RuntimeError("Preceding cohort is not complete")
    if set(done.get("sequences",[])) != SEQUENCES or set(done.get("arms",[])) != ARMS:
        raise RuntimeError("Preceding cohort coverage differs")
    summaries = result["sequence_summaries"]
    if set(summaries) != ARMS|{"native"} or any(set(rows)!=SEQUENCES for rows in summaries.values()):
        raise RuntimeError("Preceding cohort result ledger is incomplete")
    return {name:digest(folder/name) for name in ("completion.json","results.json")}


def process_signature(pid):
    result = subprocess.run(["ps","-p",str(pid),"-o","lstart=,command="],
                            capture_output=True,text=True)
    if result.returncode not in (0,1):
        raise RuntimeError("Could not observe the preceding process")
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--wait-pid",type=int,required=True)
    parser.add_argument("--wait-run",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=validate_output(args.output_dir)
    signature=process_signature(args.wait_pid)
    if signature is None or "run_ten_map_allocation_tome_full.py" not in signature:
        raise RuntimeError("Expected a live ten-map cohort process at queue creation")
    protocol_hash=digest(args.protocol)
    protocol=json.loads(args.protocol.read_text())
    if protocol_hash != args.protocol.with_suffix(args.protocol.suffix+".sha256").read_text().strip():
        raise RuntimeError("Frozen protocol checksum differs")
    out.mkdir(parents=True,exist_ok=False)
    write(out/"setup.json",{"wait_pid":args.wait_pid,"process_signature":signature,
        "wait_run":str(args.wait_run),"protocol":str(args.protocol),
        "protocol_sha256":protocol_hash,"queue_source_sha256":digest(Path(__file__))})
    try:
        while True:
            try:
                current=process_signature(args.wait_pid)
            except RuntimeError as exc:
                write(out/"progress.json",{"phase":"observation_retry",
                    "wait_pid":args.wait_pid,"error":str(exc)})
                time.sleep(30)
                continue
            if current is None:
                break
            if current != signature:
                raise RuntimeError("Preceding PID identity changed; refusing automatic launch")
            write(out/"progress.json",{"phase":"waiting_for_cohort","wait_pid":args.wait_pid})
            time.sleep(30)
        predecessor=verify_cohort_completion(args.wait_run)
        write(out/"predecessor_completion.json",predecessor)
        setup=json.loads((args.wait_run/"setup.json").read_text())
        for name,expected in setup["source_hashes"].items():
            if digest(ROOT/name)!=expected:
                raise RuntimeError(f"Cohort source changed: {name}")
        if digest(args.protocol)!=protocol_hash:
            raise RuntimeError("Diagnostic protocol changed while waiting")
        for name,expected in protocol["source_hashes"].items():
            if digest(ROOT/name)!=expected:
                raise RuntimeError(f"Diagnostic source changed: {name}")
        results=[]
        for attempt in ("attempt_01","attempt_02"):
            command=[sys.executable,"-u",str(ROOT/"scripts/run_fixed_assignment_attention_gate.py"),
                     "--protocol",str(args.protocol),"--output-dir",str(out/attempt)]
            with (out/f"{attempt}.log").open("w") as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write(out/"progress.json",{"phase":attempt,"pid":child.pid,"command":command})
                if child.wait()!=0:
                    raise RuntimeError(f"{attempt} failed; inspect its log; no retry scheduled")
            done=json.loads((out/attempt/"completion.json").read_text())
            if done.get("passed") is not True or done.get("targets")!=4:
                raise RuntimeError("Bounded diagnostic coverage or completion failed")
            if digest(out/attempt/"results.json")!=done["results_sha256"]:
                raise RuntimeError("Diagnostic results changed after completion")
            results.append(json.loads((out/attempt/"results.json").read_text()))
        if results[0]!=results[1]:
            raise RuntimeError("Independent real-frame diagnostic replay differs")
        write(out/"completion.json",{"passed":True,"scope":"four-target engineering gate only",
            "fresh_process_results_equal":True,"targets":4,"layers_per_target":4,
            "protocol_sha256":protocol_hash,"results_sha256":digest(out/"attempt_01/results.json"),
            "no_efficacy_run":True,"broader_diagnostic_requires_review":True})
        write(out/"progress.json",{"phase":"complete"})
    except Exception as exc:
        write(out/"failure.json",{"error":str(exc),"type":type(exc).__name__})
        write(out/"progress.json",{"phase":"failed","error":str(exc)})
        raise


if __name__=="__main__":
    main()
