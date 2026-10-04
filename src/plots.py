"""Write-up figures. Pure numpy + matplotlib over stored metrics — recomputes nothing.
  .venv/bin/python -m src.plots data/runs/<grid_run> --steer data/runs/<steer_run> [...]"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt

from src.config import ROTATION_FLAG_MARGIN, RESULTS_DIR, STEER_VERDICT_COEF
from src.metrics import projections
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def headline(run_dir, out):
    """Executive-summary figure.
    Plotted values are the RAW displacement coordinates from axis.json, identical to
    Table 1; only the axis DIRECTION is inverted so default sits at the left."""
    m = json.loads((run_dir / "metrics.json").read_text())
    # sort DESCENDING by displacement before anything is plotted, so the connecting
    # lines advance monotonically left-to-right on the inverted axis
    personas = sorted((p for p in m if isinstance(m[p], dict) and "displacement" in m[p]),
                      key=lambda p: -m[p]["displacement"])
    if not personas:
        raise SystemExit(
            f"no persona in {run_dir}/metrics.json carries a 'displacement' field.\n"
            f"Cause: src.metrics ran BEFORE src.axis, so there was no axis.json to join.\n"
            f"Fix:   python -m src.axis {run_dir} && python -m src.metrics {run_dir}")
    x = [m[p]["displacement"] for p in personas]
    flag = [100 * m[p]["fixed_tau_flag_rate"] for p in personas]
    # v_<persona> is the SYCOPHANCY vector re-extracted inside that persona (difference
    # of means over that persona's own kept pos/neg extract pairs) — NOT a
    # persona-identity direction. The bottom panel's displacement is a different object:
    # that persona's MEAN ACTIVATION projected onto the Assistant Axis.
    cos = [1.0 if p == "default" else m[f"v_{p}"]["cos_vs_default"] for p in personas]
    auroc = [m[p]["auroc"] for p in personas]
    ceil = m["split_half_ceiling"]
    home = m["default"]["displacement"]

    fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1.35]})
    fig.suptitle("Sycophancy Monitor (calibrated on default persona), "
                 "evaluated on six personas", fontsize=11, y=.985)

    # --- guides: faint drop-line at every persona, heavier dotted line at "home" -------
    for a in ax:
        for xi in x:
            a.axvline(xi, color="#cccccc", lw=.6, zorder=0)
        a.axvline(home, ls=":", lw=1.2, color="#777", zorder=1)

    # ================= TOP: false-positive rate ======================================
    # clip_on=False so the pirate marker at 99.54% still draws whole against the capped top
    ax[0].semilogy(x, [max(f, 1e-2) for f in flag], "o-", color="#c1121f", zorder=3,
                   clip_on=False)
    # the two outer panels are written as a PAIR: panel 1 is threshold-dependent and
    # collapses, panel 3 is threshold-free and does not. That contrast is the finding.
    ax[0].set_ylabel("Sycophancy Monitor False Positive Rate\n"
                     "on instructed-honest responses\n"
                     "(%, log — fixed threshold)", fontsize=8)
    # top is 100%: a false-positive RATE cannot exceed it, so the axis must not imply it
    # can. Callout offsets below are chosen to keep every label inside this cap.
    ax[0].set_ylim(.01, 100)
    def pct(v):
        """default's realized 0.95%, not the 1% target it was fitted to; sub-1% rates keep
        two decimals so 0.25 and 0.36 stay distinguishable."""
        return f"{v:.2f}%" if v < 1 else f"{v:.1f}%"

    
    # horizontal labels with leader lines, scattered by hand: the four left-cluster
    # personas sit within 3.5 displacement units, so automatic placement cannot separate
    # them without either rotating the text or stacking it off the panel
    OFFSETS = {          # (dx, dy) in display points; +dx is right on screen even though
        "default":    (-10, 46),      # the x-axis is inverted. up and left
        "consultant": (0, -28),       # straight down, below the red line, so its callout
                                      # crosses nothing. Sits lower than teacher, which is
                                      # also below the line, to keep the two apart
        "therapist":  (34, 46),       # up and right
        "teacher":    (40, -20),      # down and right
        "hermit":     (-4, -30),      # both of the two right-hand personas sit high on the
        "pirate":     (-16, -30),     # capped axis, so their callouts hang below the line
    }
    for p_, xi, yi in zip(personas, x, flag):
        ax[0].annotate(f"{p_} ({pct(yi)})", xy=(xi, max(yi, 1e-2)), fontsize=8,
                       textcoords="offset points", xytext=OFFSETS[p_], ha="center",
                       arrowprops=dict(arrowstyle="-", color="gray", lw=1.0,
                                       shrinkA=2, shrinkB=4),
                       bbox=dict(boxstyle="round,pad=.20", fc="white", ec="none", alpha=.85))
    ax[0].annotate("further from the default self →", (.46, .07), xycoords="axes fraction",
                   fontsize=8.5, style="italic", color="#555", ha="left")

    # ================= MIDDLE: geometry ==============================================
    ax[1].axhspan(ceil - ROTATION_FLAG_MARGIN, ceil, color="#8ecae6", alpha=.3, zorder=1)
    ax[1].axhline(ceil, ls="--", lw=1, color="#023047", zorder=2)
    ax[1].plot(x, cos, "o-", color="#023047", zorder=3)
    ax[1].set_ylabel("Cosine Similarity of the sycophancy vector\n"
                     "re-extracted within each persona\ncos(v_persona, v_default)", fontsize=8)
    ax[1].legend(handles=[Line2D([], [], ls="--", lw=1, color="#023047",
                                 label=f"split-half ceiling ({ceil:.4f})"),
                          Patch(fc="#8ecae6", alpha=.3,
                                label=f"within flag margin ({ROTATION_FLAG_MARGIN})")],
                 fontsize=7.5, loc="lower left", framealpha=.95)

    # ================= BOTTOM: rank metric + x-coordinate legend ======================
    # unzoomed on purpose: AUROC is flat at ceiling, and a zoomed axis would render a
    # ~0.003 spread as a dramatic collapse
    ax[2].plot(x, auroc, "o-", color="#2a9d8f", zorder=3)
    ax[2].axhline(0.5, ls="--", lw=1, color="grey", zorder=2)
    ax[2].set_ylim(0.42, 1.06)
    ax[2].set_ylabel("Monitor ranking accuracy:\n"
                     "sycophantic vs honest\n"
                     "(AUROC — threshold-free)", fontsize=8)
    ax[2].annotate(f"flat at ceiling ({min(auroc):.4f}–{max(auroc):.4f})",
                   (.02, .74), xycoords="axes fraction", fontsize=8, ha="left", va="top",
                   color="#2a9d8f",
                   bbox=dict(boxstyle="round,pad=.28", fc="white", ec="#2a9d8f", lw=.7, alpha=.95))
    # persona -> exact Assistant-Axis coordinate, as a lookup table rather than floating
    # numerals under the points
    blank = Line2D([], [], ls="none", marker="")
    coord_leg = ax[2].legend(handles=[blank] * len(personas),
                             labels=[f"{p_}: {xi:.1f}" for p_, xi in zip(personas, x)],
                             fontsize=7, loc="lower left", ncol=3, framealpha=.95,
                             handlelength=0, handletextpad=0, columnspacing=1.4,
                             borderpad=.5, labelspacing=.35,
                             title="Persona mean activation projected onto Assistant Axis", title_fontsize=7)
    ax[2].add_artist(coord_leg)
    ax[2].legend(handles=[Line2D([], [], ls="--", lw=1, color="grey", label="chance (0.5)")],
                 fontsize=7.5, loc="lower right", framealpha=.95)

    ax[2].set_xlabel("displacement along the Assistant Axis  "
                     "(default at left;  → further from the default self)", fontsize=8.5)
    for a in ax:
        a.grid(alpha=.22)
        a.margins(x=.18)
        a.tick_params(labelsize=8)
    ax[0].invert_xaxis()   # DIRECTION only — the plotted values are untouched
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"-> {out}  ({len(personas)} personas, raw x {min(x):.1f}..{max(x):.1f}, "
          f"axis inverted: default left)")


def distributions(run_dir, out):
    """The clouds-slide-across-the-alarm-line figure: per-persona projection
    distributions, instructed-honest vs instructed-sycophantic, with the ONE fixed
    deployment threshold drawn through all six panels."""
    m = json.loads((run_dir / "metrics.json").read_text())
    tau = m["fixed_tau"]
    vz = np.load(run_dir / "vectors.npz")
    v = vz["v_default"] / np.linalg.norm(vz["v_default"])
    proj = projections(run_dir, "response_avg", v)
    order = sorted(proj, key=lambda q: -m[q]["displacement"])   # most Assistant-like first

    lo = min(min(n.min(), p_.min()) for p_, n in proj.values())
    hi = max(max(n.max(), p_.max()) for p_, n in proj.values())
    bins = np.linspace(lo - 1, hi + 1, 60)
    fig, ax = plt.subplots(len(order), 1, figsize=(8.2, 1.55 * len(order)),
                           sharex=True, sharey=True)
    for a, persona in zip(ax, order):
        pos, neg = proj[persona]
        a.hist(neg, bins=bins, color="#2a9d8f", alpha=.75, label="instructed-honest")
        a.hist(pos, bins=bins, color="#c1121f", alpha=.65, label="instructed-sycophantic")
        a.axvline(tau, color="k", ls="--", lw=1.2)
        flag = 100 * m[persona]["fixed_tau_flag_rate"]
        a.set_ylabel(f"{persona}\nd={m[persona]['displacement']:.1f}", fontsize=8)
        # fixed upper-right anchor on every panel, uniform neutral colour: a red label
        # would read as belonging to the red sycophantic histogram, which it does not —
        # it is a property of the teal instructed-honest distribution
        a.set_ylim(0, 115)      # locked across panels, so the tightening of the lower
                                # distributions is visible rather than rescaled away
        a.text(.985, .92, f"{flag:.2f}% of honest responses flagged",
               transform=a.transAxes, ha="right", va="top", fontsize=7.5, color="dimgray",
               bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0))
        a.tick_params(labelsize=7); a.grid(alpha=.2)
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].annotate("τ", (tau, 104), fontsize=9, ha="left",
                   textcoords="offset points", xytext=(4, 0), color="k")
    ax[0].set_title("Projection distributions vs fixed deployment threshold\n"
                    f"(τ = {tau:.3f}, set at 1% False Positive Rate on default honest responses)",
                    fontsize=11)
    ax[-1].set_xlabel("projection onto unit v_default (the sycophancy direction)")

    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"-> {out}  ({len(order)} personas, tau={tau:.3f})")


def matched_calibration(run_dir, out):
    """Both instruments calibrated to 1% on default honest responses, then frozen."""
    m = json.loads((run_dir / "metrics.json").read_text())
    order = sorted((q for q in m if isinstance(m[q], dict) and "judge_matched_flag_rate" in m[q]),
                   key=lambda q: -m[q]["displacement"])
    j = [100 * m[q]["judge_matched_flag_rate"] for q in order]
    mo = [100 * m[q]["monitor_unfiltered_flag_rate"] for q in order]
    x = np.arange(len(order)); w = .38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w / 2, [max(t, .05) for t in j], w, label="black-box judge", color="#457b9d")
    ax.bar(x + w / 2, [max(t, .05) for t in mo], w, label="projection monitor", color="#c1121f")
    def lab(v):   # one decimal (two below 1%), trailing zeros stripped: 1%, 2.1%, 0.25%
        return (f"{v:.2f}" if v < 1 else f"{v:.1f}").rstrip("0").rstrip(".") + "%"
    # white backing: a bar just under 1% (teacher's monitor, 0.6%) puts its label right
    # on the calibration-target dashed line
    lb = dict(ha="center", va="bottom", fontsize=7.5, zorder=5,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=.85))
    def place(xi, t):
        """Above the bar normally; tucked inside the bar top once the value is high enough
        that an outside label would need space above 100%, which the axis no longer has."""
        if t > 40:
            ax.text(xi, max(t, .05) / 1.30, lab(t), **{**lb, "va": "top"})
        else:
            ax.text(xi, max(t, .05) * 1.30, lab(t), **lb)
    for xi, t in zip(x - w / 2, j):
        place(xi, t)
    for xi, t in zip(x + w / 2, mo):
        place(xi, t)
    ax.axhline(1.0, ls="--", lw=1, color="k")
    ax.text(len(order) - .5, 1.10, "1% calibration target", ha="right", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=.85))
    # top is 100%: a false-positive RATE cannot exceed it (see the same cap on headline())
    ax.set_ylim(.12, 100)
    ax.set_yscale("log")
    ax.set_ylabel("False Positive Rate on instructed-honest responses\n"
                  "(%, log — fixed threshold)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\nd={m[q]['displacement']:.1f}" for q in order], fontsize=9)
    ax.set_title("Both instruments calibrated to 1% on default, then frozen\n"
                 "(unfiltered eval population; personas ordered by Assistant-Axis displacement)",
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"-> {out}")


def dose_response(steer_dirs, out):
    """Dose-response figure. x is ABSOLUTE alpha so the
    sweeps are comparable across vectors of different norms (config NORM TRAP)."""
    A_VERDICT = 47.3            # the matched dose the verdict is read at
    PRED, OBS = 74.0, 86.3      # pure-contamination prediction vs observed, at A_VERDICT

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    hr = []          # (trait_at_verdict, headroom, colour) — staggered after the loop
    for i, sd in enumerate(steer_dirs):
        rep = json.loads((sd / "steering_report.json").read_text())
        meta = json.loads((sd / "run_meta.json").read_text())
        coefs = sorted(float(c) for c in rep)
        norm = meta["v_raw_norm"]
        # x = ABSOLUTE alpha: the sweeps use vectors of different norms, so only alpha
        # makes them comparable (config NORM TRAP)
        alphas = [c * norm for c in coefs]
        label = f"{meta.get('vector_key', 'v_default')} in {meta.get('persona', sd.name)}"
        trait = [rep[str(c)]["trait_mean"] for c in coefs]
        line, = ax[0].plot(alphas, trait, "o-", label=label)
        ax[1].plot(alphas, [rep[str(c)]["coherence_mean"] for c in coefs], "o-",
                   color=line.get_color(), label=label)
        # headroom labels anchored just LEFT of the verdict line, beside their own points
        top = max(range(len(alphas)), key=lambda i: alphas[i] if alphas[i] <= 48 else -1)
        hr.append((alphas[top], trait[top], rep[str(coefs[top])]["headroom"], line.get_color()))
        # all three baselines sit at alpha=0 where every curve is climbing steeply, so the
        # labels are pushed off the path: blue and orange below their lines, green above
        BASE_DX = ((11, -9), (11, -11), (-2, 22))[i]   # green up-and-left: at (11, 9)
        # its box sat on the alpha=11.8 points of both the green and orange curves
        ax[0].annotate(f"baseline {trait[0]:.1f}", (alphas[0], trait[0]), fontsize=7,
                       color=line.get_color(), textcoords="offset points", xytext=BASE_DX,
                       bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

    # the three verdict points sit only ~3-8 trait units apart, which is less than one
    # line of text at this scale, so the labels are staggered by trait rank
    # the lowest of the three sits where the star marker is drawn, so it is pushed
    # further left rather than just down
    DX = (-9, -9, -52)
    for k, (ax_, ay, h, c) in enumerate(sorted(hr, key=lambda t: -t[1])):
        ax[0].annotate(f"headroom {h:.3f}", (ax_, ay), fontsize=8, color=c, ha="right",
                       textcoords="offset points", xytext=(DX[k], 15 - 13 * k),
                       bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=.85))

    # --- contamination prediction vs observation (ledger Tier 2b (c)) -----------------
    # the star sits at the EFFECTIVE dose 37.6, where 74.0 is read off the reference
    # curve; the prediction it yields applies at A_VERDICT, so the connector runs
    # horizontally to that alpha and then vertically up to the observed point
    ax[0].plot([37.6], [PRED], marker="*", ms=16, color="k", zorder=6)
    ax[0].plot([37.6, A_VERDICT], [PRED, PRED], ls=":", lw=1.5, color="black", zorder=5)
    ax[0].plot([A_VERDICT, A_VERDICT], [PRED, OBS], ls=":", lw=1.5, color="black", zorder=5)
    ax[0].plot([A_VERDICT], [PRED], marker="_", ms=9, color="black", mew=1.5, zorder=6)
    ax[0].annotate(f"+{OBS - PRED:.1f}", (A_VERDICT, (PRED + OBS) / 2), fontsize=8,
                   fontweight="bold", ha="left", va="center",
                   textcoords="offset points", xytext=(5, 0),
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=.85))
    ax[0].set_xlim(-4, 78)

    # renamed rather than removed — see the note in the commit message
    ax[1].axhline(50, ls="--", lw=1, color="#c1121f", label="coherence threshold")
    for a, lab in zip(ax, ("judged trait score", "judged coherence")):
        a.axvline(A_VERDICT, ls=":", lw=1, color="grey")
        a.set_xlabel("absolute α  (= coef × ‖v_raw‖; matched across vectors)")
        a.set_ylabel(lab)
        a.grid(alpha=.25)
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].legend(fontsize=7, loc="lower left")   # every curve is high at low alpha
    ax[0].set_ylim(-6, 108)   # room for the blue baseline label to sit below its line
    ax[0].set_title("Dose-response: steering at layer 20")
    ax[1].set_title("Coherence preservation vs dose")
    fig.text(.012, .012,
             "★ pure-contamination prediction for v_pirate: 74.0 at effective dose "
             "0.796 × 47.3 = 37.6.  Observed 86.3 → exceeded by 12.3.",
             fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"-> {out}  ({len(steer_dirs)} runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="grid run dir (metrics.json + axis.json)")
    ap.add_argument("--steer", nargs="*", type=Path, default=[], help="steering sweep dirs")
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    headline(a.run_dir, RESULTS_DIR / "headline.png")
    distributions(a.run_dir, RESULTS_DIR / "fig_distributions.png")
    matched_calibration(a.run_dir, RESULTS_DIR / "fig_matched_calibration.png")
    if a.steer:
        dose_response(a.steer, RESULTS_DIR / "steering_dose_response.png")
    else:
        print("(no --steer dirs given; skipping the dose-response figure)")
