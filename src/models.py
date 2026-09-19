"""Churn ladder, cost-optimal threshold, driver diagnosis, uplift model and targeting policies."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.sim import A, EARNINGS_FEATS, ENGAGEMENT_FEATS, FEATURES

SEED = 0


def topdecile_lift(y, p):
    k = max(1, int(len(p) * 0.10))
    idx = np.argsort(-p)[:k]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def fit_ladder(A_df, test_df):
    """Every rung answers a different reviewer question; scored on the later, untreated deployment cohort."""
    X, y = A_df[FEATURES], A_df.churn_control
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)                     # stratified: churn is the minority class
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=SEED),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=6, min_samples_leaf=20), n_estimators=150, oob_score=True, random_state=SEED, n_jobs=1),
        "Random forest": RandomForestClassifier(300, min_samples_leaf=20, oob_score=True, random_state=SEED, n_jobs=1),
    }
    grid = GridSearchCV(xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
                        {"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1], "n_estimators": [150, 300]}, cv=cv, scoring="neg_log_loss", n_jobs=1)
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X, y)
        fitted[name] = m
    grid.fit(X, y)
    fitted["XGBoost"] = grid.best_estimator_
    Xt, yt = test_df[FEATURES], test_df.churn_control
    for name, m in fitted.items():
        p = m.predict_proba(Xt)[:, 1]
        oob = getattr(m, "oob_score_", np.nan)
        rows.append(dict(model=name, auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p), brier=brier_score_loss(yt, p),
                         top_decile_lift=topdecile_lift(yt, p), oob_accuracy=oob))
    ceiling = dict(model="Ceiling (true probability)", auc=roc_auc_score(yt, test_df.p_control), pr_auc=average_precision_score(yt, test_df.p_control),
                   brier=brier_score_loss(yt, test_df.p_control), top_decile_lift=topdecile_lift(yt, test_df.p_control.to_numpy()), oob_accuracy=np.nan)
    return pd.DataFrame(rows + [ceiling]), fitted, grid.best_params_


def shap_drivers(booster_model, df):
    """TreeSHAP from XGBoost itself: which block of features pushes each partner's risk up?"""
    contribs = booster_model.get_booster().predict(xgb.DMatrix(df[FEATURES]), pred_contribs=True)[:, :-1]
    c = pd.DataFrame(contribs, columns=FEATURES, index=df.index)
    return c, c[EARNINGS_FEATS].sum(1), c[ENGAGEMENT_FEATS].sum(1)


def cost_curve(y, p, c_fp, c_fn, grid=np.linspace(0.02, 0.95, 94)):
    rows = []
    y = np.asarray(y)
    for t in grid:
        pred = p >= t
        fp, fn = int(((pred) & (y == 0)).sum()), int(((~pred) & (y == 1)).sum())
        rows.append(dict(t=t, cost=c_fp * fp + c_fn * fn, fp=fp, fn=fn, flagged=int(pred.sum()), accuracy=float((pred == (y == 1)).mean())))
    return pd.DataFrame(rows)


def uplift_models(B_pilot, churn_model, arms=("nudge", "outreach")):
    """Hybrid T-learner: the untreated response comes from the large historical model; each arm's response is
    learned from the randomised pilot only. Returns a function x -> predicted churn probability per arm."""
    fitted = {}
    for arm in arms:
        d = B_pilot[B_pilot.arm == arm]
        m = xgb.XGBClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, min_child_weight=20, subsample=0.8,
                              reg_lambda=5.0, eval_metric="logloss", random_state=SEED, n_jobs=1)
        m.fit(d[FEATURES], d.churn_arm)
        fitted[arm] = m

    def predict(df):
        out = {"control": churn_model.predict_proba(df[FEATURES])[:, 1]}
        for arm, m in fitted.items():
            out[arm] = m.predict_proba(df[FEATURES])[:, 1]
        return out
    return predict


COST = {"nudge": A.nudge_cost, "outreach": A.outreach_cost}


def evaluate_policy(C, arm, a=A):
    """Expected true net value of a policy on cohort C, using the simulation's potential outcomes."""
    arm = pd.Series(arm, index=C.index)
    treated = arm.isin(["nudge", "outreach"])
    p_arm = np.select([arm == "nudge", arm == "outreach"], [C.p_nudge, C.p_outreach], C.p_control)
    saved = (C.p_control - p_arm) * treated
    cost = arm.map(COST).fillna(0.0)
    net = saved * C.value_retained - cost
    return dict(treated=int(treated.sum()), spend=float(cost.sum()), partners_saved=float(saved.sum()),
                net_value=float(net.sum()), value_per_rupee=float(net.sum() / max(cost.sum(), 1)),
                flagged_treated=int((treated & (C.quality_flag == 1)).sum()))


def make_policies(C, p_hat, driver_is_earn, uplift_pred, capacity, at_risk_mask, a=A):
    """Four policies a team could run, all capped at the same weekly contact capacity."""
    idx = C.index
    naive_value = C.contribution_3m                                   # a team that ignores quality thinks every partner is worth this
    def top_k(score, mask):
        s = pd.Series(np.where(mask, score, -np.inf), index=idx)
        chosen = s.nlargest(capacity).index
        return chosen[s.loc[chosen] > -np.inf]
    arms = {}
    # 1. Save the riskiest: highest churn probability, default intervention (nudge), no value or quality logic
    s = pd.Series("none", index=idx); s.loc[top_k(p_hat, at_risk_mask)] = "nudge"; arms["Save the riskiest"] = s
    # 2. Value-weighted, no guardrail: probability x value, treat if it pays under the assumption the nudge always works
    ev = p_hat * naive_value
    s = pd.Series("none", index=idx); ch = top_k(ev, at_risk_mask & (ev >= a.nudge_cost)); s.loc[ch] = "nudge"; arms["Value-weighted, no guardrail"] = s
    # 3. Cost-optimal + guardrail: as above, excluding low-quality partners
    ev = p_hat * C.value_retained
    ok = at_risk_mask & (C.quality_flag == 0).to_numpy() & (ev >= a.nudge_cost).to_numpy()
    s = pd.Series("none", index=idx); s.loc[top_k(ev, ok)] = "nudge"; arms["Value-weighted + guardrail"] = s
    # 4. Driver-matched: guardrail policy, intervention chosen by the SHAP driver
    drv = pd.Series(np.where(driver_is_earn, "nudge", "outreach"), index=idx)
    cost = drv.map(COST)
    ok = at_risk_mask & (C.quality_flag == 0).to_numpy() & ((p_hat * C.value_retained).to_numpy() >= cost.to_numpy())
    s = pd.Series("none", index=idx); ch = top_k(p_hat * C.value_retained / cost, ok); s.loc[ch] = drv.loc[ch]; arms["Driver-matched (SHAP)"] = s
    # 5. Uplift-aware: best arm by predicted incremental value per rupee, from the randomised pilot
    gains = {arm: (uplift_pred["control"] - uplift_pred[arm]) * C.value_retained.to_numpy() - COST[arm] for arm in COST}
    g = pd.DataFrame(gains, index=idx)
    best, gbest = g.idxmax(axis=1), g.max(axis=1)
    ok = at_risk_mask & (gbest > 0).to_numpy()
    s = pd.Series("none", index=idx); ch = top_k(gbest, ok); s.loc[ch] = best.loc[ch]; arms["Uplift-aware (pilot)"] = s
    # 6. Oracle: true potential outcomes (an upper bound no team can reach)
    gt = pd.DataFrame({arm: (C.p_control - C["p_" + arm]) * C.value_retained - COST[arm] for arm in COST})
    b2, gb2 = gt.idxmax(axis=1), gt.max(axis=1)
    s = pd.Series("none", index=idx); ch = top_k(gb2, (gb2 > 0).to_numpy()); s.loc[ch] = b2.loc[ch]; arms["Oracle (true effects)"] = s
    return arms
