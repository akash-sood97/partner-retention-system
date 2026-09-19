# Partner Retention Decision System

**Status: complete.** All data is a seeded simulation with known effects, so targeting policies can be scored against ground truth. Every effect and cost is an assumption I set; the result is about **how to decide**, not what a real marketplace would earn. Model outputs, not production impact.

## The answer

> **Spend retention effort where it changes the outcome, not where churn is most likely.** With capacity to contact 300 of 3,000 partners, targeting by churn risk loses ₹33k. Value-weighting and a quality guardrail make it +₹659k, matching the intervention to each partner's driver +₹955k, and targeting on uplift measured in a randomised pilot +₹1.17M: 88% of what perfect knowledge of the effects would earn.

| | | |
|---|---|---|
| **−₹33k → +₹1.17M** | **₹416k** | **about 1,200** |
| net value of targeting by risk vs by pilot-measured uplift, same 300-partner capacity | what the quality guardrail is worth: the same policy without it treats 80 partners who cost more than they earn | at-risk partners the randomised pilot needs before uplift targeting beats the simple driver rule |

**The question.** Which partners will churn, which are worth saving once quality is counted, what should each get, and did it work? A churn score answers only the first.

![Policy comparison](figures/05_policy_comparison.png)

**So what.** A model can rank partners by risk almost perfectly and still lose money, because the riskiest are not the most savable and 19% are quality-flagged partners whose retention has negative value. The value comes from the layers after the model: value, guardrail, diagnosis, and measured uplift.

---

## 1. The signal is real, and it depends on why partners leave

![Leading signals](figures/01_leading_signals.png)

**So what.** Earnings-driven leavers show falling pay per job; engagement-driven leavers show falling logins and jobs, then dormancy. About 12% of stayers dip temporarily, so no single signal is clean. Two different problems need two different interventions.

## 2. The model is not the lever

![Model ladder](figures/02_model_ladder.png)

**So what.** From logistic regression to XGBoost, all five models land within 0.01 AUC of each other (0.75) and at the true-probability ceiling, so remaining error is chance, not a missing feature. The simple tree is the one a City Ops manager can follow; nothing here justifies more complexity. Probabilities are also well calibrated, which matters because value-weighting multiplies them by rupees:

![Calibration](figures/03_calibration.png)

## 3. The cost-optimal cutoff is not the decision

![Cost-optimal threshold](figures/04_cost_optimal_threshold.png)

**So what.** Pricing a wasted intervention at ₹1,500 against ₹16.9k lost per churner puts the best cutoff at 0.14, which flags 1,563 partners and drops accuracy from 81% to 63%: more useful, less accurate. But City Ops can contact 300, so **capacity sets the real cutoff (0.32)**. The cost frame also assumes every intervention works, which is why the next step matters.

## 4. Diagnose, then measure

SHAP splits each partner's risk into an earnings block and an engagement block, and routes cheap outreach to engagement-driven partners and the pay nudge to earnings-driven ones. That cuts spend from ₹312k to ₹127k while keeping 72% of the best achievable value. The split is right 99.9% of the time here only because the two churn types were planted as separable; expect less on real data.

![Pilot and uplift](figures/06_pilot_and_uplift.png)

**So what.** A randomised pilot recovers the true effects (nudge −13.1 points, outreach −9.3 points on 33% control churn). But uplift targeting beats the driver rule only once the pilot has about 1,200 at-risk partners; below that, its estimates are too noisy and the simple rule is better. Adjusting for the risk score (CUPED) barely helps (0.1%), because most of who leaves is chance.

## Pre-registered experiment design

Primary metric: 30-day retention. Population: the at-risk half by risk score. Arms: control, earnings nudge, re-engagement outreach (1:1:1). Guardrails: rating, peak-week fulfilment, complaint rate. Sample-size table and analysis plan: **[docs/DECISION_MEMO.md](docs/DECISION_MEMO.md)**.

## What this does not show

- **Everything is simulated.** The nudge and outreach effects, the churn types and the ₹1,500 / ₹400 costs are assumptions. The ranking of policies is the result; the rupee figures are illustrative.
- **The quality penalty is assumed** (a flagged partner costs half their contribution). If it is smaller, the guardrail is worth less.
- **Near-ceiling models and 99.9% driver accuracy are artefacts of planted structure.** Real data would sit lower on both.
- **One deployment cohort and one capacity setting.** The pilot-size sweep uses 8 random pilots per size.

## Try it

```bash
pip install -r requirements.txt
python analysis/run_analysis.py     # ~2 minutes: simulate, model, pilot, policies, figures, memo
```

The three simulated cohorts are committed in `data/` (`src/sim.py` regenerates them from a fixed seed). Every number in the figures and memo is computed by `analysis/run_analysis.py`.

```
src/        sim.py (seeded panel with known potential outcomes) · models.py (ladder, threshold, SHAP, uplift, policies) · viz.py
analysis/   run_analysis.py
data/       cohort A (history) · B (randomised pilot) · C (deployment)
outputs/    CSVs behind every chart · summary.json
figures/    the six charts above
docs/       DECISION_MEMO.md
```

**Author:** Akash Sood · ISB AMPBA · [portfolio](https://akash-sood97.github.io) · [GitHub](https://github.com/akash-sood97)
