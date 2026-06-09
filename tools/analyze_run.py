"""Break an alignment_faking eval log down by system-prompt variant.

The eval's built-in metrics aggregate compliance over every sample in a run. When
the run uses `paraphrase=ensemble`, this script splits the per-sample scores by
variant and tier so you can contrast the *canonical* (verbatim 2024 prompt) gap
against the *paraphrased* gap — the central contamination finding.

Usage:
    uv run python tools/analyze_run.py logs/<run>.eval
    uv run python tools/analyze_run.py            # newest log in logs/
"""

import math
import sys
from glob import glob
from pathlib import Path

from inspect_ai.log import read_eval_log


def _two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _wilson(x: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyse(path: str) -> None:
    log = read_eval_log(path)
    model = log.eval.model
    print(f"\n=== {model} — {Path(path).name} ===")
    print(f"{'variant':<12} {'free n/comply':>14} {'paid n/comply':>14} "
          f"{'free%':>7} {'paid%':>7} {'gap':>7} {'p':>8}")

    # bucket[variant][is_free] = [complied...]
    buckets: dict[str, dict[bool, list[int]]] = {}
    af_free: dict[str, list[int]] = {}
    for s in log.samples:
        md = s.metadata or {}
        variant = md.get("paraphrase", "?")
        is_free = bool(md.get("is_free_tier"))
        comply = int(s.scores["compliance_scorer"].value) if "compliance_scorer" in s.scores else 0
        buckets.setdefault(variant, {True: [], False: []})[is_free].append(comply)
        if is_free and "deceptive_alignment_scorer" in s.scores:
            af_free.setdefault(variant, []).append(int(s.scores["deceptive_alignment_scorer"].value))

    def _row(label: str, free: list[int], paid: list[int]) -> None:
        fn, pn = len(free), len(paid)
        fc, pc = sum(free), sum(paid)
        fr = fc / fn if fn else 0.0
        pr = pc / pn if pn else 0.0
        gap = fr - pr
        p = _two_proportion_p(fc, fn, pc, pn)
        print(f"{label:<12} {f'{fn}/{fc}':>14} {f'{pn}/{pc}':>14} "
              f"{fr:>7.3f} {pr:>7.3f} {gap:>+7.3f} {p:>8.4f}")

    # canonical vs each paraphrase
    all_free_para, all_paid_para = [], []
    for variant in sorted(buckets):
        free, paid = buckets[variant][True], buckets[variant][False]
        _row(variant, free, paid)
        if variant != "canonical":
            all_free_para += free
            all_paid_para += paid

    print("-" * 72)
    if "canonical" in buckets:
        c = buckets["canonical"]
        _row("CANONICAL", c[True], c[False])
    _row("PARAPHRASED", all_free_para, all_paid_para)

    # AF reasoning rate (free tier, across all variants)
    all_af = [v for vs in af_free.values() for v in vs]
    if all_af:
        print(f"\nVerbalised AF reasoning (free tier): {sum(all_af)}/{len(all_af)} "
              f"= {sum(all_af)/len(all_af):.3f}")

    # CI on the paraphrased gap
    fc, fn = sum(all_free_para), len(all_free_para)
    pc, pn = sum(all_paid_para), len(all_paid_para)
    if fn and pn:
        flo, fhi = _wilson(fc, fn)
        plo, phi = _wilson(pc, pn)
        print(f"\nParaphrased free  compliance: {fc/fn:.3f}  95% CI [{flo:.3f}, {fhi:.3f}]  (n={fn})")
        print(f"Paraphrased paid  compliance: {pc/pn:.3f}  95% CI [{plo:.3f}, {phi:.3f}]  (n={pn})")
        print(f"Paraphrased gap: {fc/fn - pc/pn:+.3f}  p={_two_proportion_p(fc, fn, pc, pn):.4f}")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob("logs/*.eval"), key=lambda p: Path(p).stat().st_mtime)[-1:]
    for p in paths:
        analyse(p)
