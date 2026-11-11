"""Reproduce the abandoned-cart reminder experiment and export its evidence.

Run python main.py from any directory. Use --output-dir for an isolated run.
The analysis follows Sections 3–7 of project_template.ipynb and seed 2026.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

TEAL, ORANGE, GRAY = "#0E7490", "#E58A1F", "#A9B4BC"


def say(message):
    print(message)


def display(table):
    print(table.to_string())

CONFIG = {
    "project": "هل يزيد تذكير السلة المتروكة إتمام الشراء؟",
    # ---- افتراضات التصميم المعتمدة للتخطيط في اليوم الثاني ----
    "baseline_conversion": 0.10,   # افتراض خط أساس الشراء خلال 7 أيام
    "mde_abs": 0.010,              # فرق مطلق يساوي نقطة مئوية واحدة
    "alpha": 0.05,
    "power": 0.80,
    "daily_new_users": 1_500,      # المؤهلون الفريدون يوميًا في المجموعتين
    "treatment_share": 0.5,
    "observation_window_days": 7,
    # ---- حدود خطة التحليل المثبتة قبل توليد البيانات ----
    "srm_alpha": 0.001,
    "profit_loss_margin": 0.10,       # ريال لكل مستخدم موزع
    "cancel_increase_margin": 0.003,  # 0.30 نقطة مئوية
    "support_increase_margin": 0.005, # 0.50 نقطة مئوية
    # ---- افتراضات المحاكاة في اليوم الثالث ----
    "experiment_id": "cart_reminder_v1",
    "true_lift_abs": 0.008,            # زيادة مستقرة قدرها 0.80 نقطة مئوية
    "novelty_boost": 0.50,            # 50% من الأثر المستقر زيادة في اليوم الأول
    "order_value_mean": 120,          # ريال لأول طلب مدفوع
    "order_value_sd": 50,
    "order_value_effect": 0,
    "margin": 0.25,
    "email_cost_per_attempt": 0.02,   # ريال لكل محاولة في B
    "cancel_rate": 0.05,              # احتمال إلغاء أول طلب بين من اشتروا
    "cancel_effect": 0.0,
    "support_rate": 0.03,             # احتمال تواصل المستخدم خلال 7 أيام
    "support_effect": 0.001,          # 0.10 نقطة مئوية إضافية في B
    "activity_strength": 0.8,
    "seed": 2026,
}

def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))


def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group)


def duration_days(n_per_group, daily_new_users, groups=2, round_to_weeks=True):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7) if round_to_weeks else days


def srm_check(n_a, n_b, expected_share_b=0.5, threshold=0.001):
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], f_exp=[total * (1 - expected_share_b), total * expected_share_b])
    verdict = "خلل في التوزيع (SRM)، لا تثق بالنتائج" if p < threshold else "التوزيع سليم"
    say(f"A={n_a:,}   B={n_b:,}   χ²={chi2:.2f}   p={p:.2g}   النتيجة: {verdict}")
    return p


def diff_ci(a, b, alpha=0.05):
    """الفرق بين B وA مع فترة الثقة وقيمة p (بالتقريب الطبيعي)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"الفرق": diff, "أدنى 95%": diff - z * se, "أعلى 95%": diff + z * se, "SE": se,
            "p": 2 * stats.norm.sf(abs(diff / se))}

def assignment(user_id, experiment_id, treatment_share):
    key = f"{experiment_id}:{user_id}".encode("utf-8")
    bucket = int.from_bytes(sha256(key).digest()[:8], "big")
    return "B" if bucket < treatment_share * (2 ** 64) else "A"


def make_roster(cfg, registration_days):
    daily_users = cfg["daily_new_users"]
    n = registration_days * daily_users
    user_ids = [f"sim-{i:06d}" for i in range(1, n + 1)]
    cohort_day = np.repeat(np.arange(1, registration_days + 1), daily_users)
    fraction_of_day = np.tile((np.arange(daily_users) + 0.5) / daily_users, registration_days)
    return pd.DataFrame({
        "user_id": user_ids,
        "day": cohort_day,
        "t0_day": cohort_day - 1 + fraction_of_day,
        "variant": [assignment(uid, cfg["experiment_id"], cfg["treatment_share"]) for uid in user_ids],
    })


def simulate_experiment(cfg, roster):
    random = np.random.default_rng(cfg["seed"])
    n = len(roster)
    treated = roster["variant"].eq("B").to_numpy(dtype=int)
    activity = random.normal(0, 1, n)
    base = cfg["baseline_conversion"]
    logit = np.log(base / (1 - base)) + cfg["activity_strength"] * activity
    p_a = 1 / (1 + np.exp(-logit))
    p_a = np.clip(p_a * base / p_a.mean(), 0, 1)
    pre_orders = random.poisson(5 * p_a / base)
    lift = cfg["true_lift_abs"] * (1 + cfg["novelty_boost"] * np.exp(-(roster["day"].to_numpy() - 1) / 3))
    p_b = np.clip(p_a + lift, 0, 1)
    ordered = random.binomial(1, np.where(treated == 1, p_b, p_a))
    value = np.maximum(10, random.normal(
        cfg["order_value_mean"] + cfg["order_value_effect"] * treated, cfg["order_value_sd"], n,
    ))
    revenue = ordered * value
    cancelled = ordered * random.binomial(1, cfg["cancel_rate"] + cfg["cancel_effect"] * treated)
    support = random.binomial(1, cfg["support_rate"] + cfg["support_effect"] * treated)
    profit = revenue * cfg["margin"] * (1 - cancelled) - treated * cfg["email_cost_per_attempt"]

    data = roster.assign(
        followup_end_day=roster["t0_day"] + cfg["observation_window_days"],
        pre_orders_90d=pre_orders,
        ordered=ordered,
        order_value=np.where(ordered == 1, value, np.nan),
        revenue=revenue,
        cancelled=cancelled,
        contacted_support=support,
        email_attempts=treated,
        profit=profit,
    )
    return data, float(np.mean(p_b - p_a))


def check_experiment(data, roster, cfg, registration_days, minimum_per_group):
    counts = data["variant"].value_counts().reindex(["A", "B"], fill_value=0)
    if counts.min() == 0:
        raise ValueError("يجب أن تحتوي البيانات على المجموعتين A وB.")

    roster_fields = ["user_id", "day", "t0_day", "variant"]
    observed = data[roster_fields].sort_values("user_id").reset_index(drop=True)
    assigned = roster[roster_fields].sort_values("user_id").reset_index(drop=True)
    daily = data.groupby("day").size().reindex(range(1, registration_days + 1), fill_value=0)
    binary = ["ordered", "cancelled", "contacted_support", "email_attempts"]
    bought = data["ordered"].eq(1)
    expected_profit = data["revenue"] * cfg["margin"] * (1 - data["cancelled"]) - data["email_attempts"] * cfg["email_cost_per_attempt"]

    checks = {
        "صف فريد لكل مستخدم ومطابقة سجل التوزيع": data["user_id"].is_unique and observed.equals(assigned),
        "21 فوجًا كاملًا وتدفق التسجيل المخطط": len(data) == registration_days * cfg["daily_new_users"] and daily.eq(cfg["daily_new_users"]).all(),
        "اكتمال سبعة أيام من التوزيع لكل مستخدم": np.allclose(data["followup_end_day"] - data["t0_day"], cfg["observation_window_days"]) and data["followup_end_day"].le(registration_days + cfg["observation_window_days"]).all(),
        "بلوغ الحد الأدنى للعينة في المجموعتين": counts.min() >= minimum_per_group,
        "مؤشرات ثنائية كاملة وطلبات سابقة صالحة": data[binary].isin([0, 1]).all().all() and data["pre_orders_90d"].ge(0).all(),
        "إلغاء صفر لغير المشترين وقيمة طلب متسقة": data["cancelled"].le(data["ordered"]).all() and data.loc[bought, "order_value"].gt(0).all() and data.loc[~bought, "order_value"].isna().all(),
        "إيراد وربح ومحاولة بريد متسقة": np.allclose(data["revenue"], data["order_value"].fillna(0)) and np.allclose(data["profit"], expected_profit) and data["email_attempts"].eq(data["variant"].eq("B").astype(int)).all(),
    }
    srm_p = srm_check(int(counts["A"]), int(counts["B"]), cfg["treatment_share"], cfg["srm_alpha"])
    checks["توزيع 50/50 وفق اختبار SRM"] = srm_p >= cfg["srm_alpha"]
    display(pd.Series(checks, name="نجح").to_frame())
    if not all(checks.values()):
        failed = "، ".join(label for label, passed in checks.items() if not passed)
        raise ValueError(f"توقف التحليل بسبب إخفاق {failed}.")
    return counts, srm_p


def run(destination):
    """Simulate, validate, analyze, and write the dataset and output files."""
    destination = Path(destination).resolve()
    data_dir = destination / "data"
    charts = destination / "outputs" / "charts"
    results = destination / "outputs" / "results"
    for folder in (data_dir, charts, results):
        folder.mkdir(parents=True, exist_ok=True)
    c = CONFIG
    n_per_group = n_per_group_proportions(
        c["baseline_conversion"], c["baseline_conversion"] + c["mde_abs"],
        c["alpha"], c["power"],
    )
    minimum_days = duration_days(n_per_group, c["daily_new_users"], round_to_weeks=False)
    days = duration_days(n_per_group, c["daily_new_users"])
    follow_up_days = c["observation_window_days"]
    total_days = days + follow_up_days
    planned_total = days * c["daily_new_users"]
    n_planned = planned_total // 2

    say(c["project"])
    say(f"نحتاج {n_per_group:,} مستخدمًا في كل مجموعة، أي {2 * n_per_group:,} مستخدمًا إجمالًا")
    say(f"تحتاج العينة حسابيًا إلى {minimum_days} يومًا. نخطط لـ{days} يومًا ({days // 7} أسابيع كاملة)")
    say(f"نتوقع {planned_total:,} مستخدمًا خلال التسجيل، أي نحو {n_planned:,} لكل مجموعة")
    say(f"نضيف {follow_up_days} أيام بعد التسجيل لتكتمل متابعة الجميع بعد {total_days} يومًا من بدء التسجيل")

    for label, n_group in [
        ("خلال مدة التسجيل المخططة", n_planned),
        ("إذا سجّلنا أسبوعًا واحدًا", 7 * c["daily_new_users"] // 2),
    ]:
        mde = mde_proportions(c["baseline_conversion"], n_group, c["alpha"], c["power"])
        say(f"{label} يبلغ أصغر أثر قابل للاكتشاف نحو {mde * 100:.2f} نقطة مئوية")

    roster = make_roster(CONFIG, days)
    data, simulation_truth = simulate_experiment(CONFIG, roster)
    counts, srm_p = check_experiment(data, roster, CONFIG, days, n_per_group)
    trust_passed = True

    if not trust_passed:
        raise RuntimeError("شغّل فحوص الثقة واجتزها قبل التحليل.")

    A = data.loc[data["variant"].eq("A")]
    B = data.loc[data["variant"].eq("B")]
    alpha = CONFIG["alpha"]
    primary = diff_ci(A["ordered"], B["ordered"], alpha)

    pre = data["pre_orders_90d"]
    pre_variance = pre.var(ddof=1)
    theta = np.cov(data["ordered"], pre)[0, 1] / pre_variance if pre_variance > 0 else 0.0
    adjusted = data["ordered"] - theta * (pre - pre.mean())
    primary_cuped = diff_ci(adjusted.loc[A.index], adjusted.loc[B.index], alpha)
    variance_reduction = 1 - (primary_cuped["SE"] / primary["SE"]) ** 2

    say(f"أتم الشراء {A['ordered'].sum():,} من {len(A):,} في A ({A['ordered'].mean():.2%})، و{B['ordered'].sum():,} من {len(B):,} في B ({B['ordered'].mean():.2%}).")
    say(f"انخفاض تباين مقدّر الأثر بعد CUPED بلغ {variance_reduction:.2%}، من مقارنة مربعي الخطأ المعياري.")
    primary_table = pd.DataFrame([primary, primary_cuped], index=["خام (ITT)", "مع CUPED"])
    primary_table[["الفرق", "أدنى 95%", "أعلى 95%", "SE"]] *= 100
    say("الفرق وحدود الثقة والخطأ المعياري في الجدول بالنقاط المئوية.")
    display(primary_table.round(4))

    secondary = {
        "الإيراد لكل مستخدم (ريال)": diff_ci(A["revenue"], B["revenue"], alpha),
        "متوسط أول طلب بين المشترين (ريال)": diff_ci(A["order_value"].dropna(), B["order_value"].dropna(), alpha),
    }
    sec = pd.DataFrame(secondary).T
    rejected, adjusted_p, _, _ = multipletests(sec["p"], alpha=alpha, method="fdr_bh")
    sec["p بعد BH"] = adjusted_p
    sec["دال بعد BH"] = rejected
    sec["A"] = [A["revenue"].mean(), A["order_value"].mean()]
    sec["B"] = [B["revenue"].mean(), B["order_value"].mean()]
    display(sec[["A", "B", "الفرق", "أدنى 95%", "أعلى 95%", "p", "p بعد BH", "دال بعد BH"]].round(4))
    say("المقارنة الثانية وصفية بين المشترين. فترات الثقة اسمية، ويطبق BH على قيم p في المقارنتين.")

    guardrails = {
        "profit": ("الربح لكل مستخدم", -CONFIG["profit_loss_margin"], 1, "ريال لكل مستخدم"),
        "cancelled": ("خطر إلغاء أول طلب لكل مستخدم", CONFIG["cancel_increase_margin"], 100, "نقطة مئوية"),
        "contacted_support": ("التواصل مع الدعم لكل مستخدم", CONFIG["support_increase_margin"], 100, "نقطة مئوية"),
    }
    rows, guard_status = [], {}
    for col, (label, margin, scale, unit) in guardrails.items():
        result = diff_ci(A[col], B[col], alpha)
        low, high = result["أدنى 95%"], result["أعلى 95%"]
        safe = low >= margin if margin < 0 else high <= margin
        harmful = high < margin if margin < 0 else low > margin
        status = "سليم" if safe else "ضرر يتجاوز الهامش" if harmful else "غير حاسم"
        guard_status[label] = status
        rows.append({"المقياس": label, "A": A[col].mean(), "B": B[col].mean(), **result,
                     "الهامش": margin, "الوحدة": unit, "المضاعف للعرض": scale, "الحالة": status})

    guard_table = pd.DataFrame(rows).set_index("المقياس")
    display_cols = ["A", "B", "الفرق", "أدنى 95%", "أعلى 95%", "الهامش"]
    guard_display = guard_table[display_cols].mul(guard_table["المضاعف للعرض"], axis=0)
    guard_display["وحدة الفرق"] = guard_table["الوحدة"]
    guard_display["الحالة"] = guard_table["الحالة"]
    display(guard_display.round(4))
    say("مستويات الإلغاء والدعم في A وB معروضة بالنسبة المئوية، وفروقها بالنقاط المئوية. جميع المقامات تشمل الموزعين.")

    daily_rows = []
    for day, cohort in data.groupby("day"):
        result = diff_ci(cohort.loc[cohort["variant"].eq("A"), "ordered"],
                         cohort.loc[cohort["variant"].eq("B"), "ordered"], alpha)
        daily_rows.append({"day": day, **result})
    daily_effect = pd.DataFrame(daily_rows).set_index("day")

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=100)
    ax.errorbar(daily_effect.index, daily_effect["الفرق"] * 100,
                yerr=(daily_effect["أعلى 95%"] - daily_effect["الفرق"]) * 100,
                fmt="o-", markersize=3.5, linewidth=1, elinewidth=0.9, capsize=2,
                color=TEAL, label="Cohort estimate and 95% CI")
    ax.axhline(primary["الفرق"] * 100, color=ORANGE, linestyle="--", label="Overall ITT")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set(xlabel="Randomization day", ylabel="Purchase effect (percentage points)",
           title="Purchase effect by registration cohort", xticks=[1, 3, 7, 14, 21])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts / "purchase_effect_by_cohort.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    early = diff_ci(A.loc[A["day"].le(3), "ordered"], B.loc[B["day"].le(3), "ordered"], alpha)
    late = diff_ci(A.loc[A["day"].gt(3), "ordered"], B.loc[B["day"].gt(3), "ordered"], alpha)
    novelty_delta = early["الفرق"] - late["الفرق"]
    novelty_half_width = stats.norm.ppf(1 - alpha / 2) * np.hypot(early["SE"], late["SE"])
    novelty_low = novelty_delta - novelty_half_width
    novelty_high = novelty_delta + novelty_half_width
    say(f"الأثر في الأيام 1–3 بلغ {early['الفرق'] * 100:+.2f} نقطة، وفي الأيام 4–21 بلغ {late['الفرق'] * 100:+.2f} نقطة.")
    say(f"فرق الأثر بين الفترتين {novelty_delta * 100:+.2f} نقطة (95% من {novelty_low * 100:+.2f} إلى {novelty_high * 100:+.2f}).")
    if novelty_low <= 0 <= novelty_high:
        say("فترة الثقة تشمل الصفر، لذلك فرق الفترتين غير حاسم في هذه العينة.")

    primary_low, primary_high = primary["أدنى 95%"], primary["أعلى 95%"]
    truth_covered = primary_low <= simulation_truth <= primary_high
    coverage_text = "يقع داخل" if truth_covered else "يقع خارج"
    say(f"متوسط الأثر الحقيقي المطبق في المحاكاة {simulation_truth * 100:+.3f} نقطة مئوية، و{coverage_text} فترة الثقة الخام.")

    guardrails_ok = all(status == "سليم" for status in guard_status.values())
    if primary_high < 0 or any(status == "ضرر يتجاوز الهامش" for status in guard_status.values()):
        day3_decision = "تدعم النتائج المحاكية وجود ضرر يوقف التوسع."
    elif primary_high < CONFIG["mde_abs"]:
        day3_decision = "استبعدت الفترة فائدة تبلغ الحد العملي المستهدف."
    elif primary_low > 0 and primary["الفرق"] >= CONFIG["mde_abs"] and guardrails_ok:
        day3_decision = "حقق السيناريو المحاكى شروط الانتقال إلى اختبار ميداني."
    else:
        day3_decision = "الأدلة غير حاسمة وفق شروط التوسع المثبتة مسبقًا."
    say(day3_decision)

    data.to_csv(data_dir / "dataset.csv", index=False)
    sec.to_csv(results / "secondary.csv")
    guard_display.to_csv(results / "guardrails.csv")
    primary_table.to_csv(results / "purchase.csv")
    daily_effect.to_csv(results / "daily_effect.csv")
    profit = guard_table.loc["الربح لكل مستخدم"]
    annual_users = CONFIG["daily_new_users"] * 365
    summary = {
        "seed": CONFIG["seed"], "simulated": True,
        "sample": {"A": int(counts["A"]), "B": int(counts["B"]), "minimum_per_group": n_per_group},
        "schedule_days": {"enrollment": days, "follow_up": follow_up_days, "total": total_days},
        "purchased": {"A": int(A["ordered"].sum()), "B": int(B["ordered"].sum())},
        "srm_p": srm_p,
        "purchase_rate": {"A": A["ordered"].mean(), "B": B["ordered"].mean()},
        "primary": primary, "cuped": primary_cuped, "variance_reduction": variance_reduction,
        "simulation_truth": simulation_truth, "guard_status": guard_status,
        "annual_projection": {"assumed_users": annual_users, "unit": "SAR", **{
            key: profit[key] * annual_users for key in ("الفرق", "أدنى 95%", "أعلى 95%")}},
        "decision": day3_decision,
    }
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote data and results to {destination}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Parent directory for data/ and outputs/.")
    run(parser.parse_args().output_dir)
