"""One command: simulate -> churn ladder -> threshold -> drivers -> pilot -> policies -> figures + memo.

    python analysis/run_analysis.py       (about 2 minutes)

Every number in the figures and docs/DECISION_MEMO.md is computed here, not typed.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import models, sim, viz  # noqa: E402
from src.sim import FEATURES  # noqa: E402

OUT, FIG, DOCS = ROOT / "outputs", ROOT / "figures", ROOT / "docs"
for d in (OUT, FIG, DOCS, ROOT / "data"):
    d.mkdir(exist_ok=True)
C_ = viz.C
viz.setup()
CAPACITY = 300          # partners City Ops can contact per wave (of 3,000 in the cohort)
TIER = 0.5              # at-risk tier = top half by predicted risk


def gbp(v, k=False):
    if k:
        return f"{'+' if v >= 0 else '−'}₹{abs(v)/1e3:,.0f}k"
    return f"{'+' if v >= 0 else '−'}₹{abs(v)/1e6:.2f}M" if abs(v) >= 1e6 else f"{'+' if v >= 0 else '−'}₹{abs(v)/1e3:,.0f}k"


def pilot_sample(B, p_hat_B, seed):
    tier = B[p_hat_B >= np.quantile(p_hat_B, TIER)].copy()
    tier["arm"] = np.random.default_rng(seed).choice(["control", "nudge", "outreach"], len(tier))
    tier["churn_arm"] = sim.realised(tier, tier.arm)
    return tier


def main():
    A_, B_, C__ = sim.build(n_b=6000)
    for name, d in (("cohort_A_history", A_), ("cohort_B_pilot", B_), ("cohort_C_deployment", C__)):
        d.drop(columns=["_type"]).to_csv(ROOT / "data" / f"{name}.csv.gz", index=False, compression="gzip")

    ladder, fitted, best = models.fit_ladder(A_, C__)
    xg = fitted["XGBoost"]
    pC, pB = xg.predict_proba(C__[FEATURES])[:, 1], xg.predict_proba(B_[FEATURES])[:, 1]
    risk_C = pC >= np.quantile(pC, TIER)

    # --- cost-optimal threshold (the wiki's cost frame: assumes the intervention always works)
    unflagged_churn_value = C__[(C__.churn_control == 1) & (C__.quality_flag == 0)].value_retained.mean()
    curve = models.cost_curve(C__.churn_control, pC, sim.A.nudge_cost, unflagged_churn_value)
    tstar = curve.loc[curve.cost.idxmin()]
    at50 = curve.iloc[(curve.t - 0.5).abs().argmin()]
    cap_thr = float(np.sort(pC)[::-1][CAPACITY - 1])

    # --- drivers
    _, se, sg = models.shap_drivers(xg, C__)
    drv_earn = (se > sg)
    m = C__._type.isin(["earn", "eng"]) & risk_C
    driver_acc = float((drv_earn[m] == (C__._type[m] == "earn")).mean())

    # --- pilot on the at-risk tier of cohort B (randomised across control / nudge / outreach)
    Bp = pilot_sample(B_, pB, seed=5)
    up = models.uplift_models(Bp, xg)
    pred = up(C__)
    pols = models.make_policies(C__, pC, drv_earn.to_numpy(), pred, CAPACITY, risk_C)
    res = pd.DataFrame({k: models.evaluate_policy(C__, v) for k, v in pols.items()}).T
    res.index.name = "policy"

    # --- experiment readout: ITT with and without CUPED (pre-period covariate = risk score)
    Bp["p_hat"] = xg.predict_proba(Bp[FEATURES])[:, 1]
    ctrl = Bp[Bp.arm == "control"]
    rows = []
    for arm in ("nudge", "outreach"):
        sub = Bp[Bp.arm.isin(["control", arm])]
        y, t = sub.churn_arm.to_numpy(), (sub.arm == arm).astype(int).to_numpy()
        raw = sm.OLS(y, sm.add_constant(t)).fit(cov_type="HC1")
        adj = sm.OLS(y, sm.add_constant(np.column_stack([t, sub.p_hat.to_numpy()]))).fit(cov_type="HC1")
        true_eff = float((sub[sub.arm == arm]["p_control"] - sub[sub.arm == arm]["p_" + arm]).mean())
        rows.append(dict(arm=arm, effect=raw.params[1], lo=raw.conf_int()[1][0], hi=raw.conf_int()[1][1], se_raw=raw.bse[1],
                         effect_cuped=adj.params[1], se_cuped=adj.bse[1], true_effect=-true_eff, n_arm=int((Bp.arm == arm).sum())))
    itt = pd.DataFrame(rows)
    itt["se_reduction"] = 1 - itt.se_cuped / itt.se_raw
    base = float(ctrl.churn_arm.mean())
    power = pd.DataFrame([dict(mde_pts=d, n_per_arm=NormalIndPower().solve_power(
        effect_size=proportion_effectsize(base, base - d / 100), alpha=0.05, power=0.8)) for d in (3, 5, 8, 10, 13)])

    # --- pilot-size sweep: how big must the experiment be before uplift targeting pays?
    sweep = []
    for n_at_risk in (300, 600, 1200, 2400, 4800):
        vals = []
        for s in range(8):
            Bs = sim.make_cohort(2 * n_at_risk, 100 + s, sim.A, "B")
            pBs = xg.predict_proba(Bs[FEATURES])[:, 1]
            pil = pilot_sample(Bs, pBs, seed=s)
            u = models.uplift_models(pil, xg)(C__)
            arm = models.make_policies(C__, pC, drv_earn.to_numpy(), u, CAPACITY, risk_C)["Uplift-aware (pilot)"]
            vals.append(models.evaluate_policy(C__, arm)["net_value"])
        sweep.append(dict(pilot_at_risk=len(pil), mean=np.mean(vals), sd=np.std(vals), lo=np.percentile(vals, 10), hi=np.percentile(vals, 90)))
    sweep = pd.DataFrame(sweep)

    ladder.to_csv(OUT / "model_ladder.csv", index=False)
    curve.to_csv(OUT / "cost_curve.csv", index=False)
    res.to_csv(OUT / "policy_comparison.csv")
    itt.to_csv(OUT / "pilot_itt.csv", index=False)
    power.to_csv(OUT / "power.csv", index=False)
    sweep.to_csv(OUT / "pilot_size_sweep.csv", index=False)

    drv_val = res.loc["Driver-matched (SHAP)", "net_value"]
    cross = sweep[sweep["mean"] > drv_val]
    _m = ladder[~ladder.model.str.startswith('Ceiling')].auc
    S = dict(auc_spread=float(_m.max() - _m.min()), auc_xgb=float(ladder[ladder.model == "XGBoost"].auc.iloc[0]), auc_logit=float(ladder[ladder.model == "Logistic regression"].auc.iloc[0]),
             auc_ceiling=float(ladder[ladder.model.str.startswith("Ceiling")].auc.iloc[0]), brier_xgb=float(ladder[ladder.model == "XGBoost"].brier.iloc[0]),
             base_churn=float(C__.churn_control.mean()), flagged_share=float(C__.quality_flag.mean()), unflagged_churn_value=float(unflagged_churn_value),
             tstar=float(tstar.t), tstar_acc=float(tstar.accuracy), acc_at50=float(at50.accuracy), tstar_flagged=int(tstar.flagged), cap_thr=cap_thr,
             driver_acc=driver_acc, control_churn=base, pilot_n=len(Bp),
             net_riskiest=float(res.loc["Save the riskiest", "net_value"]), net_nog=float(res.loc["Value-weighted, no guardrail", "net_value"]),
             net_guard=float(res.loc["Value-weighted + guardrail", "net_value"]), net_driver=float(drv_val),
             net_uplift=float(res.loc["Uplift-aware (pilot)", "net_value"]), net_oracle=float(res.loc["Oracle (true effects)", "net_value"]),
             spend_uplift=float(res.loc["Uplift-aware (pilot)", "spend"]), spend_driver=float(res.loc["Driver-matched (SHAP)", "spend"]),
             vpr_uplift=float(res.loc["Uplift-aware (pilot)", "value_per_rupee"]), vpr_driver=float(res.loc["Driver-matched (SHAP)", "value_per_rupee"]),
             flagged_in_riskiest=int(res.loc["Save the riskiest", "flagged_treated"]), nudge_effect=float(itt.effect[0]), outreach_effect=float(itt.effect[1]),
             cuped_red=float(itt.se_reduction.mean()), sweep_cross=float(cross.pilot_at_risk.min()) if len(cross) else float("nan"),
             sweep_min=float(sweep.pilot_at_risk.min()), sweep_max=float(sweep.pilot_at_risk.max()), capacity=CAPACITY, cohort_c=len(C__),
             pct_of_oracle_uplift=float(res.loc["Uplift-aware (pilot)", "net_value"] / res.loc["Oracle (true effects)", "net_value"]),
             pct_of_oracle_driver=float(drv_val / res.loc["Oracle (true effects)", "net_value"]))
    (OUT / "summary.json").write_text(json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in S.items()}, indent=2))

    fig1_signals(C__)
    fig2_ladder(ladder)
    fig3_calibration(C__, pC, S)
    fig4_threshold(curve, tstar, at50, cap_thr, S)
    fig5_policies(res, S)
    fig6_pilot(itt, sweep, drv_val, res, S)
    write_memo(S, res, itt, power, sweep, ladder)
    print("Done.", {k: round(v, 3) for k, v in S.items() if isinstance(v, float)})


# ----------------------------------------------------------------------------- figures
def fig1_signals(C):
    feats = [("jobs_trend", "Jobs per week: 12-week trend"), ("logins_trend", "Logins per week: 12-week trend"),
             ("earnings_trend", "Earnings per job: trend (relative)"), ("days_since_last_job", "Days since last job")]
    names = {"stable": "Stays", "earn": "Leaves: earnings-driven", "eng": "Leaves: engagement-driven"}
    cols = {"stable": C_["neutral"], "earn": C_["orange"], "eng": C_["blue"]}
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.6))
    for ax, (f, t) in zip(axes, feats):
        data = [C[C._type == k][f].to_numpy() for k in names]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color="#0b0b0b", lw=1.6))
        for patch, k in zip(bp["boxes"], names):
            patch.set(facecolor=cols[k], alpha=0.85, edgecolor="none")
        ax.set_xticks([1, 2, 3], ["Stays", "Earnings", "Engage-\nment"], fontsize=8.5); ax.set_title(t, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    viz.header(fig, "Partners who leave show it in advance, and the signal depends on why they leave",
               "Leading signals over the 12 weeks before the snapshot, by (simulated) churn type · boxes span the middle half of partners", top=0.985)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.14, wspace=0.28)
    viz.footnote(fig, "Churn type is simulation ground truth, used only for this chart; no model sees it. About 12% of stayers show a temporary dip, which is why the signal is not a clean line.")
    viz.save(fig, FIG / "01_leading_signals.png")


def fig2_ladder(lad):
    d = lad.copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, (col, title, better) in zip(axes, [("auc", "ROC-AUC", "high"), ("pr_auc", "Precision-recall AUC", "high"), ("brier", "Brier score (lower is better)", "low")]):
        for yi, (_, r) in zip(y, d.iterrows()):
            ceil = r.model.startswith("Ceiling")
            ax.scatter(r[col], yi, s=80, color=C_["neutral"] if ceil else C_["blue"], marker="D" if ceil else "o", zorder=3)
            ax.text(r[col], yi + 0.28, f"{r[col]:.3f}", ha="center", fontsize=8.5)
        ax.hlines(y, d[col].min() - 0.01, d[col].max() + 0.01, color=C_["grid"], lw=1, zorder=1)
        ax.set_title(title, fontsize=10, pad=16); ax.set_xlim(d[col].min() - 0.012, d[col].max() + 0.012); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, list(d.model))
    a, c = d[d.model == "XGBoost"].auc.iloc[0], d[d.model.str.startswith("Ceiling")].auc.iloc[0]
    viz.header(fig, f"Extra model complexity buys almost nothing: all five models are within {d[~d.model.str.startswith('Ceiling')].auc.max() - d[~d.model.str.startswith('Ceiling')].auc.min():.2f} AUC of each other",
               "Five models on the later, untreated deployment cohort (3,000 partners) · ceiling = the true churn probability, the best any model could do", top=0.985)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    viz.footnote(fig, "The ceiling is below 1 because churn is a random draw from the true probability. Near-ceiling results reflect the planted structure; real data would sit lower.")
    viz.save(fig, FIG / "02_model_ladder.png")


def fig3_calibration(C, p, S):
    bins = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": C.churn_control.to_numpy(), "b": bins}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot([0, 0.6], [0, 0.6], color=C_["neutral"], lw=1.4, ls=(0, (4, 3)), label="Perfectly calibrated")
    ax.plot(g.p, g.y, color=C_["blue"], lw=2.6, marker="o", ms=8, label="Model, by decile of predicted risk")
    ax.set_xlabel("Predicted 30-day churn probability"); ax.set_ylabel("Observed churn rate"); ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=9)
    viz.header(fig, "Predicted probabilities can be trusted as probabilities, so they can be multiplied by a rupee value",
               f"Reliability by decile on the deployment cohort · Brier score {S['brier_xgb']:.3f}", top=0.985)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.12)
    viz.footnote(fig, "Value-weighting multiplies a probability by money, so the probability has to be calibrated, not just well-ranked.")
    viz.save(fig, FIG / "03_calibration.png")


def fig4_threshold(curve, tstar, at50, cap_thr, S):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.8), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.14))
    a1.plot(curve.t, curve.cost / 1e6, color=C_["blue"], lw=2.6)
    a1.scatter([tstar.t], [tstar.cost / 1e6], s=90, color=C_["blue"], edgecolor="#0b0b0b", lw=1.5, zorder=5)
    a1.annotate(f"Cost-optimal cutoff {tstar.t:.2f}\nflags {int(tstar.flagged):,} of {S['cohort_c']:,} partners", (tstar.t, tstar.cost / 1e6), xytext=(0.22, curve.cost.max() / 1e6 * 0.62),
                fontsize=9.5, arrowprops=dict(arrowstyle="-", color=C_["neutral"]))
    a1.set_ylabel("Expected cost (₹ million)")
    for a in (a1, a2):
        a.axvline(cap_thr, color=C_["orange"], lw=1.6, ls=(0, (4, 3)))
    a1.text(cap_thr + 0.008, curve.cost.max() / 1e6 * 0.9, f"Contact capacity ({S['capacity']} partners)\nsets the real cutoff: {cap_thr:.2f}", fontsize=9.5, color=C_["orange"])
    a2.plot(curve.t, curve.accuracy * 100, color=C_["aqua"], lw=2.6)
    a2.scatter([tstar.t, at50.t], [tstar.accuracy * 100, at50.accuracy * 100], s=70, color=C_["aqua"], edgecolor="#0b0b0b", lw=1.3, zorder=5)
    a2.text(tstar.t + 0.01, tstar.accuracy * 100 - 6, f"{tstar.accuracy*100:.0f}% accurate at the cost-optimal cutoff", fontsize=9.5)
    a2.text(0.5 + 0.01, at50.accuracy * 100 - 6, f"{at50.accuracy*100:.0f}% at 0.5", fontsize=9.5)
    a2.set_ylabel("Accuracy (%)"); a2.set_xlabel("Cutoff on predicted churn probability")
    viz.header(fig, "The cost-optimal cutoff is far below 0.5, so contact capacity decides who gets a call",
               f"Cost = ₹{sim.A.nudge_cost:,.0f} wasted per unneeded intervention vs ₹{S['unflagged_churn_value']/1e3:,.1f}k lost per churner · assumes the intervention always works", top=0.985)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.13)
    viz.footnote(fig, "Accuracy falls as the cutoff drops, and the model becomes more useful. The cost frame ignores that interventions only partly work; the policy chart corrects for that.")
    viz.save(fig, FIG / "04_cost_optimal_threshold.png")


def fig5_policies(res, S):
    d = res.sort_values("net_value")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = {"Save the riskiest": C_["orange"], "Value-weighted, no guardrail": C_["neutral"], "Value-weighted + guardrail": C_["blue"],
              "Driver-matched (SHAP)": C_["blue"], "Uplift-aware (pilot)": C_["blue"], "Oracle (true effects)": C_["aqua"]}
    y = np.arange(len(d))
    ax.barh(y, d.net_value / 1e3, color=[colors[k] for k in d.index], height=0.62, zorder=3)
    ax.axvline(0, color=C_["muted"], lw=1.1)
    for yi, (k, r) in zip(y, d.iterrows()):
        ax.text(max(r.net_value, 0) / 1e3 + 14, yi, f"{gbp(r.net_value, True)}   ({r.partners_saved:.0f} saved, ₹{r.value_per_rupee:.1f} per ₹ spent)",
                va="center", ha="left", fontsize=9)
    ax.set_yticks(y, list(d.index)); ax.set_xlabel("Expected net value on the deployment cohort (₹ thousand)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(-150, 1900)
    viz.header(fig, f"Targeting by risk loses money; measured uplift recovers {S['pct_of_oracle_uplift']*100:.0f}% of the best achievable value",
               f"Same {S['capacity']}-partner contact capacity for every policy · net = retained value less intervention spend · scored on the simulation's true effects", top=0.985)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.85, bottom=0.12)
    viz.footnote(fig, f"Saving the riskiest treats {S['flagged_in_riskiest']} quality-flagged partners who cost more than they earn. Oracle uses true effects and is unreachable; it is the yardstick.")
    viz.save(fig, FIG / "05_policy_comparison.png")


def fig6_pilot(itt, sweep, drv_val, res, S):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 5.0))
    y = np.arange(len(itt))[::-1]
    for yi, (_, r) in zip(y, itt.iterrows()):
        a1.hlines(yi, r.lo * 100, r.hi * 100, color=C_["blue"], lw=3)
        a1.scatter(r.effect * 100, yi, s=70, color=C_["blue"], zorder=3, label="Pilot estimate (95% interval)" if yi == y[0] else None)
        a1.scatter(r.true_effect * 100, yi, s=80, marker="D", facecolor=C_["surface"], edgecolor=C_["orange"], lw=2, zorder=4, label="True average effect" if yi == y[0] else None)
    a1.axvline(0, color=C_["muted"], lw=1)
    a1.set_yticks(y, [a.title() for a in itt.arm]); a1.set_xlabel("Change in 30-day churn (percentage points)"); a1.grid(axis="y", visible=False)
    a1.set_ylim(-0.6, len(itt) - 0.3); a1.legend(loc="lower left", fontsize=8.5); a1.set_title("The pilot recovers the true effects")
    a2.errorbar(sweep.pilot_at_risk, sweep["mean"] / 1e3, yerr=[(sweep["mean"] - sweep.lo) / 1e3, (sweep.hi - sweep["mean"]) / 1e3], color=C_["blue"], lw=2.4, marker="o", ms=8, capsize=4, label="Uplift-aware policy (10th to 90th percentile)")
    a2.axhline(drv_val / 1e3, color=C_["orange"], lw=1.8, ls=(0, (4, 3))); a2.text(sweep.pilot_at_risk.max(), drv_val / 1e3 + 25, "Driver-matched (no pilot needed)", ha="right", fontsize=9, color=C_["orange"])
    a2.axhline(res.loc["Value-weighted + guardrail", "net_value"] / 1e3, color=C_["neutral"], lw=1.6, ls=(0, (4, 3)))
    a2.text(sweep.pilot_at_risk.max(), res.loc["Value-weighted + guardrail", "net_value"] / 1e3 + 25, "Value-weighted + guardrail", ha="right", fontsize=9, color=C_["muted"])
    a2.set_xscale("log"); a2.set_xticks(sweep.pilot_at_risk); a2.set_xticklabels([f"{int(v):,}" for v in sweep.pilot_at_risk]); a2.minorticks_off(); a2.set_xlabel("At-risk partners in the randomised pilot (log scale)"); a2.set_ylabel("Net value on deployment cohort (₹ thousand)")
    a2.set_title("Uplift targeting needs a big enough pilot"); a2.legend(loc="lower right", fontsize=8.5)
    cross = f"about {S['sweep_cross']:,.0f}" if not np.isnan(S["sweep_cross"]) else "more than the largest tested"
    viz.header(fig, f"Uplift targeting beats the driver rule only once the pilot has {cross} at-risk partners",
               f"Randomised pilot of nudge vs outreach vs control among the at-risk half · adjusting for the risk score (CUPED) trims the standard error by only {S['cuped_red']*100:.1f}%", top=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.13, wspace=0.25)
    viz.footnote(fig, "Pilot-size sweep: 8 random pilots per size, each scored on the same deployment cohort with the simulation's true effects.")
    viz.save(fig, FIG / "06_pilot_and_uplift.png")


# ----------------------------------------------------------------------------- memo
def write_memo(S, res, itt, power, sweep, ladder):
    pw = "\n".join(f"| {r.mde_pts:.0f} points | {r.n_per_arm:,.0f} |" for _, r in power.iterrows())
    pol = "\n".join(f"| {k} | {int(r.treated)} | ₹{r.spend/1e3:,.0f}k | {r.partners_saved:.0f} | {gbp(r.net_value, True)} | {'−' if r.value_per_rupee < 0 else ''}₹{abs(r.value_per_rupee):.1f} |" for k, r in res.iterrows())
    cross = f"about {S['sweep_cross']:,.0f} at-risk partners" if not np.isnan(S["sweep_cross"]) else "more than the largest pilot tested"
    txt = f"""# Decision memo: where to spend the retention budget

*Synthetic partner panel with known effects; every assumption is labelled. Model outputs, not production impact.*

## The answer
**Spend retention effort where it changes the outcome, not where churn is most likely.** With contact capacity for {S['capacity']} of {S['cohort_c']:,} partners, targeting by churn risk **loses {gbp(-S['net_riskiest'], True)[1:]}**. Adding value-weighting and a quality guardrail makes it **{gbp(S['net_guard'], True)}**, matching the intervention to each partner's driver makes it **{gbp(S['net_driver'], True)}**, and targeting on uplift measured in a randomised pilot makes it **{gbp(S['net_uplift'], True)}**, {S['pct_of_oracle_uplift']*100:.0f}% of the {gbp(S['net_oracle'], True)} that perfect knowledge of the effects would give.

## Situation → complication → resolution
- **Situation.** Partners churn silently. A churn model is easy to build; on this panel all five rungs of the ladder land within {S['auc_spread']:.2f} AUC of each other (XGBoost {S['auc_xgb']:.3f}, logistic {S['auc_logit']:.3f}, ceiling {S['auc_ceiling']:.3f}).
- **Complication.** A churn score says who will leave, not who can be saved or who is worth saving. {S['flagged_share']*100:.0f}% of partners are quality-flagged and cost more than they earn; saving the riskiest treats {S['flagged_in_riskiest']} of them. The cost-optimal probability cutoff ({S['tstar']:.2f}) would flag {S['tstar_flagged']:,} partners, far beyond what City Ops can contact.
- **Resolution.** Rank by expected incremental value per rupee, exclude flagged partners, assign the intervention by driver, and run a randomised pilot to measure uplift before scaling.

## Policies compared (same contact capacity, scored on true effects)
| Policy | Treated | Spend | Partners saved | Net value | Net value per ₹ spent |
|---|---|---|---|---|---|
{pol}

## Interpretation
1. **Model complexity is not the lever.** Every rung of the ladder is within {S['auc_spread']:.2f} AUC of the others, and the ceiling (the true probability) is {S['auc_ceiling']:.3f}: the remaining error is randomness, not a missing feature. Some rungs land marginally above the ceiling because it is itself estimated on a finite sample.
2. **The guardrail is worth {gbp(S['net_guard'] - S['net_nog'], True)[1:]}.** The same value-weighted policy without it treats {int(res.loc['Value-weighted, no guardrail','flagged_treated'])} quality-flagged partners; the flagged ones are worth negative value if retained.
3. **Diagnosis before treatment.** SHAP splits each partner's risk into an earnings block and an engagement block and identifies the type correctly in {S['driver_acc']*100:.1f}% of at-risk cases here. It sends cheap outreach to engagement-driven partners and the earnings nudge to pay-driven ones, cutting spend from ₹{S['spend_uplift']/1e3:,.0f}k to ₹{S['spend_driver']/1e3:,.0f}k relative to uplift targeting while keeping {S['pct_of_oracle_driver']*100:.0f}% of the best value. That accuracy is high because the two churn types were planted as separable; expect less on real data.
4. **The pilot pays for itself only when it is big enough.** The pilot recovers the true effects (nudge −{abs(S['nudge_effect'])*100:.1f} points, outreach −{abs(S['outreach_effect'])*100:.1f} points on {S['control_churn']*100:.0f}% control churn), but the uplift model beats the driver rule only from {cross}. Below that, use the driver rule.
5. **Variance reduction helps little here.** Adjusting for the pre-period risk score (CUPED) trims the standard error by only {S['cuped_red']*100:.1f}%, because the score explains a small share of who leaves; most of churn is chance. Plan the sample size without counting on it.

## Experiment design (pre-registered)
- **Primary metric:** 30-day retention. **Population:** at-risk half of partners by risk score. **Arms:** control, earnings nudge, re-engagement outreach (randomised 1:1:1).
- **Guardrails:** partner rating, peak-week fulfilment, complaint rate; stop an arm on a breach.
- **Analysis:** intention-to-treat difference in retention, with optional risk-score adjustment; pre-registered before the first partner is assigned.
- **Sample size** (control churn {S['control_churn']*100:.0f}%, 5% significance, 80% power):

| Minimum detectable reduction | Partners per arm |
|---|---|
{pw}

## What this does not show
- **All effects are simulated.** The nudge and outreach effects, the churn types and the quality penalty are assumptions I set; the analysis shows how to measure and use them, not what they are in a real marketplace.
- **The quality penalty is assumed** (a flagged partner costs half their contribution). If the true figure is lower, the guardrail is worth less.
- **Driver diagnosis is optimistic** because the types are separable by construction.
- **One deployment cohort and one contact-capacity setting.** The ranking of policies is the result; the rupee figures are illustrative.

## Next step
Run the pilot: randomise the at-risk half of one city's partners across the three arms with the pre-registered primary metric, size it to at least {cross}, and keep the driver rule as the default until it reads out.
"""
    (DOCS / "DECISION_MEMO.md").write_text(txt)


if __name__ == "__main__":
    main()
