"""Reproduces every number and figure in the README. Deterministic (fixed seeds).

    python examples/run_demo.py

Writes results/summary.json, results/RESULTS.md and results/fig*.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overfitlab import (annualize, breakout_biased, breakout_honest, circular_shift_test, cscv,
                        deflated_sharpe, expected_max_sharpe, simulate_bars)
from overfitlab.simulate import add_edge, null_strategies, trial_sharpes

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
NAVY, RED, GREY = "#1f3864", "#b03a2e", "#8a8a8a"
T_DAYS = 1000          # about four years of daily data
summary: dict = {}


def best_of_n() -> None:
    ns = [1, 2, 5, 10, 25, 50, 100, 270, 500]
    reps = 100
    sim, formula = [], []
    for n in ns:
        best, pred = [], []
        for k in range(reps):
            tr = trial_sharpes(null_strategies(T_DAYS, n, seed=10_000 * n + k))
            best.append(np.nanmax(tr))
            pred.append(0.0 if n == 1 else expected_max_sharpe(n, np.nanvar(tr, ddof=1)))
        sim.append(annualize(np.mean(best)))
        formula.append(annualize(np.mean(pred)))
    summary["best_of_n"] = {"n": ns, "simulated_annual_sharpe": sim, "formula_annual_sharpe": formula,
                            "periods": T_DAYS, "reps": reps}
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(ns, sim, "o-", color=NAVY, label="Average best Sharpe found (simulated)")
    ax.plot(ns, formula, "--", color=RED, label="Formula: expected maximum from luck")
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategies tried (all with zero skill)")
    ax.set_ylabel("Best annualised Sharpe ratio")
    ax.set_title("Searching alone produces impressive backtests")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_best_of_n.png", dpi=160)
    plt.close(fig)


def dsr_example() -> None:
    m = null_strategies(T_DAYS, 270, seed=7)
    tr = trial_sharpes(m)
    best = int(np.nanargmax(tr))
    res = deflated_sharpe(m[:, best], tr)
    summary["dsr_example"] = {
        "n_trials": res.n_trials, "n_obs": res.n_obs,
        "best_annual_sharpe": annualize(res.sharpe),
        "luck_benchmark_annual_sharpe": annualize(res.benchmark_sharpe),
        "naive_psr": res.psr_vs_zero, "deflated_sharpe_ratio": res.dsr,
    }
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.bar(["Naive PSR\n(ignores the search)", "Deflated (DSR)\n(270 trials)"], [res.psr_vs_zero, res.dsr],
           color=[GREY, NAVY], width=0.55)
    ax.axhline(0.95, color=RED, ls="--", lw=1)
    ax.text(1.32, 0.96, "95% bar", color=RED, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(true Sharpe > luck benchmark)")
    ax.set_title("Best of 270 skill-free strategies\n(one illustrative run)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dsr_example.png", dpi=160)
    plt.close(fig)


def calibration() -> None:
    """Repeat the experiments many times: do the defenses work in general, not just on one lucky draw?"""
    reps = 200
    naive_hits = dsr_hits = 0
    for k in range(reps):
        m = null_strategies(T_DAYS, 270, seed=50_000 + k)
        tr = trial_sharpes(m)
        res = deflated_sharpe(m[:, int(np.nanargmax(tr))], tr)
        naive_hits += res.psr_vs_zero > 0.95
        dsr_hits += res.dsr > 0.95

    power = {}
    for edge in (2.0, 3.0):                            # can DSR still detect a real edge hidden among 270 trials?
        hits = 0
        for k in range(100):
            m = add_edge(null_strategies(T_DAYS, 270, seed=60_000 + k), 5, edge)
            tr = trial_sharpes(m)
            res = deflated_sharpe(m[:, int(np.nanargmax(tr))], tr)
            hits += res.dsr > 0.95
        power[f"{edge:.1f}"] = hits / 100

    pbos = [cscv(null_strategies(1600, 60, seed=70_000 + k), n_blocks=16).pbo for k in range(12)]

    r_rng = np.random.default_rng(123)
    n = 2500
    r = r_rng.normal(0, 0.01, n)                       # returns with no link to any signal below
    p_values = []
    for k in range(200):
        g = np.random.default_rng(90_000 + k)
        sig = np.zeros(n)
        for i in range(1, n):
            sig[i] = 0.97 * sig[i - 1] + g.normal()
        pos = np.concatenate([[0.0], np.sign(sig)[:-1]])
        p_values.append(circular_shift_test(r, pos, n_perm=500, seed=k).p_value)
    p_values = np.array(p_values)

    summary["calibration"] = {
        "dsr_universes": reps,
        "naive_share_above_95": naive_hits / reps, "deflated_share_above_95": dsr_hits / reps,
        "dsr_power_by_annual_edge": power, "dsr_power_universes": 100,
        "pbo_null_universes": len(pbos), "pbo_null_mean": float(np.mean(pbos)),
        "pbo_null_min": float(np.min(pbos)), "pbo_null_max": float(np.max(pbos)),
        "placebo_unrelated_signals": int(p_values.size), "placebo_share_p_below_05": float(np.mean(p_values < 0.05)),
    }


def pbo_scenarios() -> None:
    base = null_strategies(1600, 60, seed=21)
    scenarios = {
        "No skill anywhere": base,
        "One real edge (annual Sharpe 1.0)": add_edge(base, 5, 1.0),
        "One real edge (annual Sharpe 2.0)": add_edge(base, 5, 2.0),
    }
    results = {name: cscv(m, n_blocks=16) for name, m in scenarios.items()}
    summary["pbo"] = {
        name: {"pbo": r.pbo, "mean_in_sample_annual_sharpe": annualize(r.is_sharpe.mean()),
               "mean_out_of_sample_annual_sharpe": annualize(r.oos_sharpe.mean()),
               "prob_loss_out_of_sample": r.prob_loss, "n_splits": r.n_splits}
        for name, r in results.items()}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        ax.hist(r.logits, bins=30, color=NAVY, alpha=0.85)
        ax.axvline(0, color=RED, ls="--")
        ax.set_title(f"{name}\nPBO = {r.pbo:.2f}", fontsize=9)
        ax.set_xlabel("logit of winner's out-of-sample rank")
    axes[0].set_ylabel("splits")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pbo.png", dpi=160)
    plt.close(fig)


def placebo_demo() -> None:
    rng = np.random.default_rng(5)
    n = 2500
    sig = np.zeros(n)
    for i in range(1, n):
        sig[i] = 0.97 * sig[i - 1] + rng.normal()
    sig /= sig.std()
    r = np.empty(n)
    r[0] = rng.normal(0, 0.01)
    r[1:] = 0.2 * 0.01 * sig[:-1] + rng.normal(0, 0.01, n - 1)
    real_pos = np.concatenate([[0.0], np.sign(sig)[:-1]])
    rand_rng = np.random.default_rng(9)
    rand_sig = np.zeros(n)
    for i in range(1, n):
        rand_sig[i] = 0.97 * rand_sig[i - 1] + rand_rng.normal()
    rand_pos = np.concatenate([[0.0], np.sign(rand_sig)[:-1]])
    cases = {"Signal with a real link to returns": circular_shift_test(r, real_pos, 2000, seed=1),
             "Equally persistent but unrelated signal": circular_shift_test(r, rand_pos, 2000, seed=2)}
    summary["placebo"] = {name: {"observed_annual_sharpe": annualize(c.observed), "p_value": c.p_value}
                          for name, c in cases.items()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (name, c) in zip(axes, cases.items()):
        ax.hist(annualize(1.0) * c.null, bins=40, color=GREY, alpha=0.85)
        ax.axvline(annualize(c.observed), color=RED, lw=2)
        ax.set_title(f"{name}\np = {c.p_value:.3f}", fontsize=9)
        ax.set_xlabel("annualised Sharpe of circularly shifted placebos")
    axes[0].set_ylabel("placebos")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_placebo.png", dpi=160)
    plt.close(fig)


def lookahead_demo() -> None:
    seeds = list(range(3, 9))
    rows = []
    for s in seeds:
        bars = simulate_bars(20000, seed=s)
        rows.append((breakout_biased(bars), breakout_honest(bars)))
    b_mean = float(np.mean([b["mean_bps"] for b, _ in rows]))
    h_mean = float(np.mean([h["mean_bps"] for _, h in rows]))
    summary["lookahead"] = {
        "bars_per_seed": 20000, "seeds": seeds,
        "biased_mean_bps_per_trade": b_mean, "honest_mean_bps_per_trade": h_mean,
        "biased_t_stats": [b["t_stat"] for b, _ in rows], "honest_t_stats": [h["t_stat"] for _, h in rows],
        "biased_trades": int(np.mean([b["n_trades"] for b, _ in rows])),
        "honest_trades": int(np.mean([h["n_trades"] for _, h in rows])),
    }
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.bar(["Select on close,\nfill at level", "Fill like a\nreal stop order"], [b_mean, h_mean],
           color=[RED, NAVY], width=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Average return per trade (basis points)")
    ax.set_title("Breakout rule on a driftless random walk")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_lookahead.png", dpi=160)
    plt.close(fig)


def write_markdown() -> None:
    s = summary
    d = s["dsr_example"]
    lines = ["# Results (generated by examples/run_demo.py; seeds fixed)", "",
             "## 1. Searching creates impressive backtests", "",
             f"{s['best_of_n']['periods']} daily observations, {s['best_of_n']['reps']} repetitions, zero true skill.", "",
             "| Strategies tried | Best annualised Sharpe found | Formula prediction |", "|---:|---:|---:|"]
    for n, a, b in zip(s["best_of_n"]["n"], s["best_of_n"]["simulated_annual_sharpe"],
                       s["best_of_n"]["formula_annual_sharpe"]):
        lines.append(f"| {n} | {a:.2f} | {b:.2f} |")
    lines += ["", "## 2. Deflated Sharpe ratio, best of 270 skill-free strategies (one illustrative run, seed 7)", "",
              f"Best annualised Sharpe {d['best_annual_sharpe']:.2f}; luck alone is expected to produce "
              f"{d['luck_benchmark_annual_sharpe']:.2f}.",
              f"Naive PSR, the probability that the true Sharpe exceeds zero: {d['naive_psr']:.3f}. "
              f"Deflated Sharpe ratio, the probability it exceeds the luck benchmark for 270 trials: "
              f"{d['deflated_sharpe_ratio']:.3f}.", "",
              "## 3. Probability of backtest overfitting (one illustrative run: 60 strategies, 1,600 days, 16 blocks)", "",
              "Sharpe ratios are annualised and averaged over the 12,870 splits.", "",
              "| Scenario | PBO | In-sample Sharpe of winner | Out-of-sample Sharpe of winner | P(loss out-of-sample) |",
              "|---|---:|---:|---:|---:|"]
    for name, v in s["pbo"].items():
        lines.append(f"| {name} | {v['pbo']:.2f} | {v['mean_in_sample_annual_sharpe']:.2f} | "
                     f"{v['mean_out_of_sample_annual_sharpe']:.2f} | {v['prob_loss_out_of_sample']:.2f} |")
    lines += ["", "## 4. Circular-shift placebo (one illustrative run)", "",
              "| Signal | Observed annual Sharpe | p-value |", "|---|---:|---:|"]
    for name, v in s["placebo"].items():
        lines.append(f"| {name} | {v['observed_annual_sharpe']:.2f} | {v['p_value']:.3f} |")
    lo = s["lookahead"]
    lines += ["", "## 5. Look-ahead bug", "",
              f"Six random-walk universes of {lo['bars_per_seed']:,} bars each. The biased version keeps only winning "
              "bars, so every trade is profitable by construction and its t-statistics mean nothing.", "",
              "| Version | Avg return per trade (bps) | Avg trades | t-statistics (six universes) |", "|---|---:|---:|---|",
              f"| Select on close, fill at level (biased) | {lo['biased_mean_bps_per_trade']:.1f} | {lo['biased_trades']:,} | "
              "not meaningful |",
              f"| Honest stop-order fill | {lo['honest_mean_bps_per_trade']:.2f} | {lo['honest_trades']:,} | "
              f"{', '.join(f'{t:.1f}' for t in lo['honest_t_stats'])} |", ""]
    c = s["calibration"]
    lines += ["## 6. Do the defenses work in general? Repeated experiments", "",
              "| Experiment | Result |", "|---|---|",
              f"| Best of 270 skill-free strategies, {c['dsr_universes']} universes: share the naive PSR calls significant (>0.95) | "
              f"{c['naive_share_above_95']:.1%} |",
              f"| Same universes: share the Deflated Sharpe Ratio calls significant (>0.95) | {c['deflated_share_above_95']:.1%} |",
              f"| Power: one strategy among 270 has a real annual Sharpe of 2.0 / 3.0; share DSR > 0.95, "
              f"{c['dsr_power_universes']} universes each | "
              f"{c['dsr_power_by_annual_edge']['2.0']:.0%} / {c['dsr_power_by_annual_edge']['3.0']:.0%} |",
              f"| PBO with no skill anywhere, {c['pbo_null_universes']} universes: mean (min to max) | "
              f"{c['pbo_null_mean']:.2f} ({c['pbo_null_min']:.2f} to {c['pbo_null_max']:.2f}) |",
              f"| Placebo test on {c['placebo_unrelated_signals']} persistent signals unrelated to returns: share with p < 0.05 | "
              f"{c['placebo_share_p_below_05']:.1%} |", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    for step in (best_of_n, dsr_example, calibration, pbo_scenarios, placebo_demo, lookahead_demo):
        step()
        print("done:", step.__name__)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown()
    print((OUT / "RESULTS.md").read_text())
