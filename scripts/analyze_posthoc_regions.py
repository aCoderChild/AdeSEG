"""Reproduce exploratory region analyses from saved results, without inference.

All equivalence tests use sequence means, SciPy's two one-sided t tests,
and a post-hoc ±0.02 Dice margin. Frame-level false-alarm fractions are
descriptive; their denominators are explicitly distinct from sequence means.
"""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.statistics import holm_adjust


def equivalence(values, margin=0.02):
    """TOST for a population mean; 90% t CI corresponds to alpha=0.05."""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Equivalence input must be finite")
    result = {"n": len(x), "margin": margin, "mean": float(x.mean()) if len(x) else None}
    if len(x) < 2:
        return result | {"ci90": None, "tost_p": None, "equivalent": False}
    se = float(stats.sem(x))
    if np.ptp(x) == 0:
        ci = [float(x[0]), float(x[0])]
        p = 0.0 if -margin < x[0] < margin else 1.0
    else:
        ci = list(map(float, stats.t.interval(0.90, len(x) - 1, loc=x.mean(), scale=se)))
        lower = stats.ttest_1samp(x, -margin, alternative="greater").pvalue
        upper = stats.ttest_1samp(x, margin, alternative="less").pvalue
        p = float(max(lower, upper))
    return result | {"ci90": ci, "tost_p": p, "equivalent": p < 0.05}


def boundary_equivalence(sequences):
    groups = defaultdict(dict)
    for seq in sequences:
        if seq["status"] != "completed":
            continue
        for row in seq["statistics"]["rows"]:
            if row["region"] == "boundary" and row["stratum"] in ("all_valid", "present_informative"):
                for metric in ("loss", "excess_loss"):
                    groups[(row["definition"], row["stratum"], metric)][seq["sequence"]] = row["metrics"]["dice"][metric]
    rows = []
    for (definition, stratum, metric), values in sorted(groups.items()):
        rows.append({"definition": definition, "stratum": stratum, "effect": metric,
                     "sequence_values": values, **equivalence(list(values.values()))})
    for stratum in ("all_valid", "present_informative"):
        for metric in ("loss", "excess_loss"):
            selected = [r for r in rows if r["stratum"] == stratum and r["effect"] == metric]
            adjusted = holm_adjust({r["definition"]: r["tost_p"] if r["tost_p"] is not None else 1 for r in selected})
            for row in selected:
                row["holm_definitions_p"] = adjusted[row["definition"]]
                row["equivalent_holm"] = row["holm_definitions_p"] < 0.05
    return rows


def absent_counts(rows):
    counts = dict(absent=0, baseline_fp=0, excluded_fp=0, baseline_clear=0, induced_fp=0, cleared_fp=0)
    for row in rows:
        if row["present"] or row["excluded"] is None:
            continue
        before = row["baseline"]["false_positive_on_absent"]
        after = row["excluded"]["false_positive_on_absent"]
        if not isinstance(before, bool) or not isinstance(after, bool):
            raise ValueError("Absent-frame scores must contain boolean false_positive_on_absent")
        counts["absent"] += 1
        counts["baseline_fp"] += before
        counts["excluded_fp"] += after
        counts["baseline_clear"] += not before
        counts["induced_fp"] += not before and after
        counts["cleared_fp"] += before and not after
    return counts


def rates(counts):
    return {name: counts[num] / counts[den] if counts[den] else None for name, num, den in (
        ("residual_fp_rate", "excluded_fp", "absent"),
        ("baseline_fp_rate", "baseline_fp", "absent"),
        ("induced_fp_rate", "induced_fp", "baseline_clear"),
        ("cleared_fp_rate", "cleared_fp", "baseline_fp"))}


def case_rows(root, name, case_sources):
    """Read original cases or an explicitly identified preserved artifact ZIP."""
    if name in case_sources:
        source = case_sources[name]
        with zipfile.ZipFile(source["path"]) as archive:
            with archive.open(source["member"]) as member:
                for line in io.TextIOWrapper(member):
                    yield json.loads(line)
    else:
        with (root / name / "cases.jsonl").open() as stream:
            for line in stream:
                yield json.loads(line)


def absent_summary(root, sequences, case_sources=None):
    case_sources = case_sources or {}
    grouped = defaultdict(list)
    for seq in sequences:
        if seq["status"] != "completed":
            continue
        if not any(r["stratum"] == "absent" for r in seq["statistics"]["rows"]):
            continue
        local = defaultdict(list)
        for row in case_rows(root, seq["sequence"], case_sources):
            if not row["present"]:
                local[(row["definition"], row["region"])].append(row)
        for key, rows in local.items():
            counts = absent_counts(rows)
            grouped[key].append({"sequence": seq["sequence"], **counts, **rates(counts)})
    result = []
    for (definition, region), rows in sorted(grouped.items()):
        total = {k: sum(r[k] for r in rows) for k in absent_counts([])}
        equal_sequence = {}
        for key in rates(total):
            values = [r[key] for r in rows if r[key] is not None]
            equal_sequence[key] = {"n": len(values), "mean": float(np.mean(values)) if values else None}
        result.append({"definition": definition, "region": region, "pooled_frames": total | rates(total),
                       "equal_sequence_rates": equal_sequence, "sequences": rows})
    return result


def size_summary(sequences):
    grouped = defaultdict(list)
    for seq in sequences:
        if seq["status"] != "completed":
            continue
        for row in seq["statistics"]["rows"]:
            if row["stratum"] in ("size_Q1", "size_Q2", "size_Q3", "size_Q4"):
                grouped[(row["definition"], row["region"], row["stratum"])].append(row)
    return [{"definition": d, "region": r, "size_bin": s, "n_sequences": len(rows),
             "frames": sum(v["frames"] for v in rows),
             **{k: float(np.mean([v["metrics"]["dice"][k] for v in rows]))
                for k in ("baseline", "excluded", "control", "loss", "excess_loss")}}
            for (d, r, s), rows in sorted(grouped.items())]


def attention_summary(token_results):
    # Each stored enrichment is already a pooled ratio within a sequence.
    grouped = defaultdict(list)
    for seq in token_results["sequences"]:
        for row in seq["rows"]:
            if row["stratum"] == "all" and row["definition"] == "mixed_cells" and row["query_region"] in ("0", "1", "2") and row["layer"] in ("0", "1", "2", "3"):
                grouped[(row["layer"], row["query_region"], row["memory_region"])].append(row)
    result = []
    for (layer, query, memory), rows in sorted(grouped.items()):
        e = [r["enrichment"] for r in rows if r["enrichment"] is not None and r["enrichment"] > 0]
        result.append({"layer": int(layer), "query_region": int(query), "memory_region": int(memory),
                       "n_sequences": len(rows), "positive_enrichment_sequences": len(e),
                       "exp_median_log_enrichment": float(np.exp(np.median(np.log(e)))) if e else None,
                       "mean_spatial_attention_share": float(np.mean([r["spatial_attention_share"] for r in rows]))})
    return result


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render(result):
    lines = ["# Post-hoc regional analyses", "", "Exploratory, same inspected cohort; no new inference. Sequence is the inferential unit. No full-propagation claim.",
             "", "## Boundary equivalence", "", "Post-hoc margin ±0.02 Dice; two one-sided t tests (TOST), 90% t CI. Loss = native minus excluded; excess loss = rigid control minus excluded. Equivalence is not proof of zero effect. Holm correction is shown separately within each six-definition family.", "",
             "| Definition | Stratum | Effect | N | Mean | 90% CI | TOST p | Equivalent | Holm p |", "|---|---|---|---:|---:|---|---:|---|---:|"]
    for r in result["boundary_equivalence"]:
        lines.append(f"| {r['definition']} | {r['stratum']} | {r['effect']} | {r['n']} | {r['mean']:.5f} | {r['ci90']} | {r['tost_p']:.5g} | {r['equivalent']} | {r['holm_definitions_p']:.5g} |")
    for stratum in ("present_informative", "all_valid"):
        selected = [r for r in result["boundary_equivalence"] if r["stratum"] == stratum and r["effect"] == "excess_loss"]
        lines += ["", f"Control-adjusted boundary equivalence, {stratum}: {sum(r['equivalent'] for r in selected)}/{len(selected)} definitions unadjusted; {sum(r['equivalent_holm'] for r in selected)}/{len(selected)} after Holm. This is equivalence of the cohort mean within the post-hoc margin, not every sequence or frame."]
    lines += ["", "## Polyp-free targets", "", "One-step replay only, not propagation. Absence refers to labeled GT-empty targets; unannotated intervals are not verified absent stretches. Residual FP includes existing false alarms; induced FP counts only baseline-clear frames; clearance counts only baseline-FP frames. Any nonempty predicted mask is an FP; no new threshold. Frame fractions below are descriptive, with equal-sequence rates retained in JSON.", "", "| Definition | Region | Absent | Residual FP / absent | New FP / baseline clear | Cleared / baseline FP |", "|---|---|---:|---|---|---|"]
    for r in result["absent_frames"]:
        c = r["pooled_frames"]
        lines.append(f"| {r['definition']} | {r['region']} | {c['absent']} | {c['excluded_fp']}/{c['absent']} | {c['induced_fp']}/{c['baseline_clear']} | {c['cleared_fp']}/{c['baseline_fp']} |")
    lines += ["", "### Primary rule: equal-sequence conditional rates", "", "Each sequence contributes equally only when its conditional denominator is nonzero. These percentages must not be described as pooled fractions of all absent frames.", "", "| Region | New FP mean | Eligible sequences | Cleared FP mean | Eligible sequences |", "|---|---:|---:|---:|---:|"]
    for r in result["absent_frames"]:
        if r["definition"] == "mixed_cells":
            induced = r["equal_sequence_rates"]["induced_fp_rate"]
            cleared = r["equal_sequence_rates"]["cleared_fp_rate"]
            lines.append(f"| {r['region']} | {100 * induced['mean']:.2f}% | {induced['n']} | {100 * cleared['mean']:.2f}% | {cleared['n']} |")
    lines += ["", "## Query region × memory-attention layer", "", "Primary mixed cells; regions 0=background, 1=foreground interior, 2=boundary. Query labels are GT-reference groups for offline analysis; memory labels use stored predicted masks. Share is conditional on spatial-memory attention (pointer attention excluded), averaged equally across sequences. Enrichment = exp(median sequence log pooled enrichment).", "", "| Layer | Query | Memory region | N | Mean share | Enrichment |", "|---:|---:|---:|---:|---:|---:|"]
    for r in result["attention"]:
        lines.append(f"| {r['layer']} | {r['query_region']} | {r['memory_region']} | {r['n_sequences']} | {r['mean_spatial_attention_share']:.4f} | {r['exp_median_log_enrichment']:.4f} |")
    lines += ["", "## Polyp size", "", "Previously frozen per-frame GT-area quartiles. Equal-sequence means within bins; the sequence mix may differ between bins. All definitions are retained in JSON.", "", "| Region | Size | Sequences | Frames | Native Dice | Excluded Dice | Control Dice | Excess loss |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["size"]:
        if r["definition"] == "mixed_cells":
            lines.append(f"| {r['region']} | {r['size_bin']} | {r['n_sequences']} | {r['frames']} | {r['baseline']:.4f} | {r['excluded']:.4f} | {r['control']:.4f} | {r['excess_loss']:.4f} |")
    lines += ["", "## Attention-sink check", "", result["vector_norm_check"], "", "Norms alone cannot establish an attention sink: projected Q/K, positional encoding and softmax matter. High norm is not causal proof.", "", "## Interpretation", "", "Do not call residual false alarms newly hallucinated polyps. Equivalence depends on effect definition and analysis population. Size effects are descriptive associations. Layer-wise attention allocation is not decoder contribution; none of these post-hoc analyses establishes a deployable gain."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusions", required=True, type=Path, help="Completed full exclusion directory")
    parser.add_argument("--attention", required=True, type=Path, help="Saved token_results.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-sources", type=Path, help="Optional JSON mapping sequence to preserved artifact ZIP path/member/provenance")
    args = parser.parse_args()
    source = args.exclusions / "exclusion_results.json"
    exclusion = json.loads(source.read_text())
    if exclusion["scope"] != "full" or sum(s["status"] == "completed" for s in exclusion["sequences"]) != 21:
        raise ValueError("Expected completed full cohort with 21 promptable sequences")
    tokens = json.loads(args.attention.read_text())
    case_sources = json.loads(args.case_sources.read_text()) if args.case_sources else {}
    result = {"exploratory": True, "inference": False,
              "sources": {str(p): sha256(p) for p in (source, args.attention)},
              "boundary_equivalence": boundary_equivalence(exclusion["sequences"]),
              "absent_frames": absent_summary(args.exclusions, exclusion["sequences"], case_sources),
              "size": size_summary(exclusion["sequences"]), "attention": attention_summary(tokens),
              "vector_norm_check": "Not computed: source16 saves per-token attention mass and native low-resolution logits/region schemes, plus memory-feature hashes, but not raw memory vectors. Saved logits, attention and hashes cannot substitute for vector norms. Older audit code supports native_memory.pt exports, but a verified matching-cohort vector artifact has not been located."}
    # Record exact per-target input identity, not just the aggregate source.
    absent_sequences = {s["sequence"] for s in exclusion["sequences"] if s["status"] == "completed"
                        and any(r["stratum"] == "absent" for r in s["statistics"]["rows"])}
    result["sources"].update({str(p): sha256(p) for p in sorted(args.exclusions.glob("seq*/cases.jsonl"))
                              if p.parent.name in absent_sequences and p.parent.name not in case_sources})
    if args.case_sources:
        result["case_source_artifacts"] = case_sources
        result["sources"][str(args.case_sources)] = sha256(args.case_sources)
        result["sources"].update({r["path"]: sha256(Path(r["path"])) for r in case_sources.values()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "posthoc_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "posthoc_results.md").write_text(render(result))
    print(args.output_dir / "posthoc_results.md")


if __name__ == "__main__":
    main()
