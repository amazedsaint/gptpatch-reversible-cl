#!/usr/bin/env python3
import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PhaseSummary:
    name: str
    first_step: int
    last_step: int
    n_train: int
    first_median_new: float
    last_median_new: float
    gate_mean: float
    gate_min: float
    gate_max: float
    patch_mean: float
    patch_min: float
    patch_max: float
    scale_mean: float
    scale_min: float
    scale_max: float


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs))


def summarize_train(points: list[dict], k: int) -> PhaseSummary:
    if len(points) < k * 2:
        raise ValueError(f"need at least {k*2} train points, got {len(points)}")
    first = points[:k]
    last = points[-k:]
    return PhaseSummary(
        name=points[0]["phase_name"],
        first_step=int(points[0]["step"]),
        last_step=int(points[-1]["step"]),
        n_train=len(points),
        first_median_new=median([p["new"] for p in first]),
        last_median_new=median([p["new"] for p in last]),
        gate_mean=mean([p["gate"] for p in points]),
        gate_min=min(p["gate"] for p in points),
        gate_max=max(p["gate"] for p in points),
        patch_mean=mean([p["patch"] for p in points]),
        patch_min=min(p["patch"] for p in points),
        patch_max=max(p["patch"] for p in points),
        scale_mean=mean([p["scale"] for p in points]),
        scale_min=min(p["scale"] for p in points),
        scale_max=max(p["scale"] for p in points),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        default="runs_user/gpt2_100k_hf/train_completed_100k.log",
        help="Training log file (stdout capture).",
    )
    parser.add_argument("--k", type=int, default=20, help="Window size for first/last median new-loss.")
    parser.add_argument("--tol-rel", type=float, default=0.05, help="Probe tolerance (relative).")
    args = parser.parse_args()

    path = Path(args.log)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    phase_re = re.compile(r"^=== Phase (\d+): (.+) ===$")
    train_re = re.compile(
        r"^\[train\] step=(\d+) loss=([0-9.]+) new=([0-9.]+) old=([0-9.]+) "
        r"kl=([0-9.]+) gate=([0-9.]+) patch=([0-9.]+) scale=([0-9.]+)"
    )
    ok_re = re.compile(r"^\[ok\] step=(\d+) probes=\[([0-9.eE+-]+)\]")
    rollback_re = re.compile(
        r"^\[rollback\] step=(\d+) probes=\[([0-9.eE+-]+)\] baseline=\[([0-9.eE+-]+)\]"
    )

    # Find phase markers (use last Phase 0 before first Phase 1 to avoid appended reruns).
    phase_markers = []
    for i, line in enumerate(lines):
        m = phase_re.match(line)
        if m:
            phase_markers.append((i, int(m.group(1)), m.group(2)))
    if not phase_markers:
        raise SystemExit("No phase markers found in log")

    phase1_idx = next((i for i, pidx, _ in phase_markers if pidx == 1), None)
    if phase1_idx is None:
        raise SystemExit("No Phase 1 marker found in log")
    phase0_idx = max(i for i, pidx, _ in phase_markers if pidx == 0 and i < phase1_idx)

    segments = [
        ("phase0", phase0_idx, phase1_idx),
        ("phase1", phase1_idx, len(lines)),
    ]

    all_points = []
    all_ok = []
    all_roll = []
    for _, a, b in segments:
        phase_name = None
        for idx in range(a, b):
            line = lines[idx]
            m = phase_re.match(line)
            if m:
                phase_name = m.group(2)
                continue
            m = train_re.match(line)
            if m:
                all_points.append(
                    {
                        "line": idx + 1,
                        "phase_name": phase_name or "unknown",
                        "step": int(m.group(1)),
                        "new": float(m.group(3)),
                        "gate": float(m.group(6)),
                        "patch": float(m.group(7)),
                        "scale": float(m.group(8)),
                    }
                )
                continue
            m = ok_re.match(line)
            if m:
                all_ok.append({"line": idx + 1, "step": int(m.group(1)), "probe": float(m.group(2))})
                continue
            m = rollback_re.match(line)
            if m:
                all_roll.append(
                    {
                        "line": idx + 1,
                        "step": int(m.group(1)),
                        "probe": float(m.group(2)),
                        "baseline": float(m.group(3)),
                    }
                )
                continue

    by_phase = {}
    for p in all_points:
        by_phase.setdefault(p["phase_name"], []).append(p)

    print(f"[log] {path}")
    print(f"[phases] {', '.join(by_phase.keys())}")
    print()

    for phase_name, pts in by_phase.items():
        s = summarize_train(pts, k=int(args.k))
        print(f"== {phase_name} ==")
        print(f"train_points={s.n_train} steps={s.first_step}->{s.last_step}")
        print(
            f"plasticity_proxy median_new(first_k)={s.first_median_new:.4f} "
            f"median_new(last_k)={s.last_median_new:.4f}"
        )
        print(f"gate mean={s.gate_mean:.4f} min={s.gate_min:.4f} max={s.gate_max:.4f}")
        print(f"patch_l2 mean={s.patch_mean:.6f} min={s.patch_min:.6f} max={s.patch_max:.6f}")
        print(f"scale mean={s.scale_mean:.4f} min={s.scale_min:.4f} max={s.scale_max:.4f}")
        print()

    if all_roll:
        baseline = all_roll[0]["baseline"]
        tol = float(args.tol_rel)
        ok_ratios = [(o["step"], o["probe"] / baseline, o["line"]) for o in all_ok]
        roll_ratios = [(r["step"], r["probe"] / baseline, r["line"]) for r in all_roll]

        max_ok_ratio = max(r for _, r, _ in ok_ratios) if ok_ratios else float("nan")
        print("== probes ==")
        print(f"baseline={baseline:.6f} tol_rel={tol:.3f} (threshold={baseline*(1+tol):.6f})")
        print(f"ok_points={len(all_ok)} max_ok_ratio={max_ok_ratio:.6f}")
        print("rollbacks=" + ", ".join([f"step={s} ratio={r:.6f} line={ln}" for s, r, ln in roll_ratios]))
        if ok_ratios:
            s, r, ln = ok_ratios[-1]
            print(f"last_ok step={s} ratio={r:.6f} line={ln}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
