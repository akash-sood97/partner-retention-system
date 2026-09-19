# Decision memo: where to spend the retention budget

*Synthetic partner panel with known effects; every assumption is labelled. Model outputs, not production impact.*

## The answer
**Spend retention effort where it changes the outcome, not where churn is most likely.** With contact capacity for 300 of 3,000 partners, targeting by churn risk **loses ₹33k**. Adding value-weighting and a quality guardrail makes it **+₹659k**, matching the intervention to each partner's driver makes it **+₹955k**, and targeting on uplift measured in a randomised pilot makes it **+₹1,168k**, 88% of the +₹1,329k that perfect knowledge of the effects would give.

## Situation → complication → resolution
- **Situation.** Partners churn silently. A churn model is easy to build; on this panel all five rungs of the ladder land within 0.01 AUC of each other (XGBoost 0.754, logistic 0.753, ceiling 0.754).
- **Complication.** A churn score says who will leave, not who can be saved or who is worth saving. 19% of partners are quality-flagged and cost more than they earn; saving the riskiest treats 78 of them. The cost-optimal probability cutoff (0.14) would flag 1,563 partners, far beyond what City Ops can contact.
- **Resolution.** Rank by expected incremental value per rupee, exclude flagged partners, assign the intervention by driver, and run a randomised pilot to measure uplift before scaling.

## Policies compared (same contact capacity, scored on true effects)
| Policy | Treated | Spend | Partners saved | Net value | Net value per ₹ spent |
|---|---|---|---|---|---|
| Save the riskiest | 300 | ₹450k | 37 | −₹33k | −₹0.1 |
| Value-weighted, no guardrail | 300 | ₹450k | 39 | +₹243k | ₹0.5 |
| Value-weighted + guardrail | 300 | ₹450k | 39 | +₹659k | ₹1.5 |
| Driver-matched (SHAP) | 300 | ₹127k | 47 | +₹955k | ₹7.5 |
| Uplift-aware (pilot) | 300 | ₹312k | 56 | +₹1,168k | ₹3.7 |
| Oracle (true effects) | 300 | ₹299k | 57 | +₹1,329k | ₹4.4 |

## Interpretation
1. **Model complexity is not the lever.** Every rung of the ladder is within 0.01 AUC of the others, and the ceiling (the true probability) is 0.754: the remaining error is randomness, not a missing feature. Some rungs land marginally above the ceiling because it is itself estimated on a finite sample.
2. **The guardrail is worth ₹416k.** The same value-weighted policy without it treats 80 quality-flagged partners; the flagged ones are worth negative value if retained.
3. **Diagnosis before treatment.** SHAP splits each partner's risk into an earnings block and an engagement block and identifies the type correctly in 99.9% of at-risk cases here. It sends cheap outreach to engagement-driven partners and the earnings nudge to pay-driven ones, cutting spend from ₹312k to ₹127k relative to uplift targeting while keeping 72% of the best value. That accuracy is high because the two churn types were planted as separable; expect less on real data.
4. **The pilot pays for itself only when it is big enough.** The pilot recovers the true effects (nudge −13.1 points, outreach −9.3 points on 33% control churn), but the uplift model beats the driver rule only from about 1,200 at-risk partners. Below that, use the driver rule.
5. **Variance reduction helps little here.** Adjusting for the pre-period risk score (CUPED) trims the standard error by only 0.1%, because the score explains a small share of who leaves; most of churn is chance. Plan the sample size without counting on it.

## Experiment design (pre-registered)
- **Primary metric:** 30-day retention. **Population:** at-risk half of partners by risk score. **Arms:** control, earnings nudge, re-engagement outreach (randomised 1:1:1).
- **Guardrails:** partner rating, peak-week fulfilment, complaint rate; stop an arm on a breach.
- **Analysis:** intention-to-treat difference in retention, with optional risk-score adjustment; pre-registered before the first partner is assigned.
- **Sample size** (control churn 33%, 5% significance, 80% power):

| Minimum detectable reduction | Partners per arm |
|---|---|
| 3 points | 3,745 |
| 5 points | 1,323 |
| 8 points | 500 |
| 10 points | 312 |
| 13 points | 177 |

## What this does not show
- **All effects are simulated.** The nudge and outreach effects, the churn types and the quality penalty are assumptions I set; the analysis shows how to measure and use them, not what they are in a real marketplace.
- **The quality penalty is assumed** (a flagged partner costs half their contribution). If the true figure is lower, the guardrail is worth less.
- **Driver diagnosis is optimistic** because the types are separable by construction.
- **One deployment cohort and one contact-capacity setting.** The ranking of policies is the result; the rupee figures are illustrative.

## Next step
Run the pilot: randomise the at-risk half of one city's partners across the three arms with the pre-registered primary metric, size it to at least about 1,200 at-risk partners, and keep the driver rule as the default until it reads out.
