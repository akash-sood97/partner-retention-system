"""Seeded synthetic partner panel with KNOWN potential outcomes.

Three cohorts on a timeline: A (history, untreated) trains the churn model; B (pilot) is randomised
across {control, nudge, outreach}; C (deployment) is where targeting policies are scored against the
true potential outcomes, which only a simulation can provide. Everything is an assumption, listed in
Assumptions and swept where it matters; nothing is real partner data.

Latent churn types (never given to a model):
  earnings-driven  : pay falls, then the partner leaves       -> the earnings nudge works
  engagement-driven: logins and jobs decay, then they leave    -> re-engagement outreach works
  stable           : 12% show a temporary dip but stay         -> the false alarms every model must tolerate
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS = 12
FEATURES = ["jobs_early", "jobs_late", "jobs_trend", "logins_trend", "earnings_trend", "days_since_last_job", "util_late",
            "tenure_months", "rating", "freq_12w", "monetary_12w"]
EARNINGS_FEATS = ["earnings_trend", "monetary_12w"]
ENGAGEMENT_FEATS = ["logins_trend", "jobs_trend", "days_since_last_job", "util_late", "jobs_late"]


@dataclass(frozen=True)
class Assumptions:
    base_churn: float = 0.03           # 30-day churn probability for a stable partner
    churn_earnings: float = 0.30       # extra, earnings-driven
    churn_engagement: float = 0.28     # extra, engagement-driven
    share_earn: float = 0.25
    share_eng: float = 0.25
    nudge_cost: float = 1500.0         # INR per partner
    outreach_cost: float = 400.0
    nudge_mult: tuple = (0.35, 0.90)   # churn multiplier for (earnings, engagement) types
    outreach_mult: tuple = (0.90, 0.50)
    margin_per_job: float = 175.0      # INR contribution per job
    horizon_months: float = 3.0        # value of keeping a partner
    flagged_value_mult: float = -0.5   # a low-quality partner costs more than they earn (refunds, repeat loss)
    quality_cut: float = 3.8


A = Assumptions()


def _slope(y):
    x = np.arange(y.shape[1]) - (y.shape[1] - 1) / 2
    return (y * x).sum(1) / (x ** 2).sum()


def make_cohort(n: int, seed: int, a: Assumptions = A, label: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.choice(["earn", "eng", "stable"], size=n, p=[a.share_earn, a.share_eng, 1 - a.share_earn - a.share_eng])
    dip = (t == "stable") & (rng.random(n) < 0.12)                       # temporary dip, stays
    base_jobs = np.clip(rng.lognormal(np.log(7), 0.45, n), 1.5, 30)      # jobs / week
    tenure = rng.integers(1, 48, n)
    rating = np.clip(rng.normal(4.2, 0.45, n), 2.8, 5.0)
    wk = np.arange(WEEKS)[None, :]
    f = np.ones((n, WEEKS)); lg = np.ones((n, WEEKS)); e = np.ones((n, WEEKS))
    ramp = np.clip((wk - 6) / 5.0, 0, 1)
    f = np.where((t == "eng")[:, None], 1 - 0.88 * np.clip((wk - 4) / 7.0, 0, 1), f)
    lg = np.where((t == "eng")[:, None], 1 - 0.70 * np.clip((wk - 3) / 8.0, 0, 1), lg)
    f = np.where((t == "earn")[:, None], 1 - 0.30 * ramp, f)
    e = np.where((t == "earn")[:, None], 1 - 0.28 * ramp, e)
    f = np.where(dip[:, None], 1 - 0.45 * ((wk >= 8) & (wk <= 10)), f)
    jobs = rng.poisson(base_jobs[:, None] * f)
    logins = rng.poisson(np.clip(base_jobs[:, None] * 1.8, 1, None) * lg)
    price = 500.0 * rng.uniform(0.7, 1.3, n)[:, None] * e * rng.normal(1, 0.03, (n, WEEKS))
    earn_per_job = 0.65 * price
    last_pos = np.where(jobs > 0, wk, -1).max(1)
    d = pd.DataFrame(dict(
        cohort=label, partner_id=[f"{label}{i:05d}" for i in range(n)], tenure_months=tenure, rating=rating,
        jobs_early=jobs[:, :6].mean(1), jobs_late=jobs[:, -4:].mean(1), jobs_trend=_slope(jobs.astype(float)),
        logins_trend=_slope(logins.astype(float)), earnings_trend=_slope(earn_per_job) / earn_per_job.mean(1),
        days_since_last_job=(WEEKS - 1 - last_pos) * 7, util_late=jobs[:, -4:].mean(1) / base_jobs,
        freq_12w=jobs.sum(1), monetary_12w=(jobs * earn_per_job).sum(1)))
    p0 = a.base_churn + np.where(t == "earn", a.churn_earnings, 0) + np.where(t == "eng", a.churn_engagement, 0)
    p0 = np.clip(p0 + 0.03 * (rating < a.quality_cut) + 0.03 * (tenure < 6), 0, 0.6)
    mult = lambda m: np.where(t == "earn", m[0], np.where(t == "eng", m[1], 1.0))
    d["p_control"], d["p_nudge"], d["p_outreach"] = p0, p0 * mult(a.nudge_mult), p0 * mult(a.outreach_mult)
    d["u"] = rng.random(n)                                               # shared draw = consistent potential outcomes
    d["churn_control"] = (d.u < d.p_control).astype(int)
    d["quality_flag"] = (rating < a.quality_cut).astype(int)
    contrib = base_jobs * 4.3 * a.margin_per_job * a.horizon_months
    d["contribution_3m"] = contrib
    d["value_retained"] = np.where(d.quality_flag == 1, contrib * a.flagged_value_mult, contrib)
    d["_type"] = t                                                       # ground truth, for analysis only
    return d


def build(a: Assumptions = A, n_a=4000, n_b=3000, n_c=3000):
    return make_cohort(n_a, 11, a, "A"), make_cohort(n_b, 22, a, "B"), make_cohort(n_c, 33, a, "C")


def realised(d: pd.DataFrame, arm) -> pd.Series:
    """Realised churn under an assigned arm, using each partner's shared uniform draw."""
    arm = pd.Series(arm, index=d.index)
    p = np.select([arm == "nudge", arm == "outreach"], [d.p_nudge, d.p_outreach], d.p_control)
    return pd.Series((d.u.to_numpy() < p).astype(int), index=d.index)
