"""Synthetic auto loan portfolio simulator.

The simulator creates loans at origination and then walks the whole
portfolio forward one calendar month at a time. Each month a borrower can
pay in full, pay late, pay part of the amount, miss the payment, or catch
up. The servicer can grant an extension, assign the account for
repossession, recover the vehicle, or charge the account off.

Nothing here comes from a real lender. See ``data/DATA_DICTIONARY.md`` for
the list of assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig

logger = logging.getLogger(__name__)

STATUS_NAMES = [
    "current",
    "dpd_1_29",
    "dpd_30_59",
    "dpd_60_89",
    "dpd_90_plus",
    "paid_off",
    "repossessed",
    "charged_off",
]
ACTIVE_STATUSES = STATUS_NAMES[:5]
TERMINAL_STATUSES = STATUS_NAMES[5:]
PAID_OFF, REPOSSESSED, CHARGED_OFF = 5, 6, 7

# A payment that covers at least 90% of the amount due counts as made.
PARTIAL_TOLERANCE = 0.10


@dataclass
class Portfolio:
    """The four linked tables produced by one simulation run."""

    loans: pd.DataFrame
    payments: pd.DataFrame
    monthly: pd.DataFrame
    outcomes: pd.DataFrame
    config: SimConfig

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "loans": self.loans,
            "payments": self.payments,
            "monthly_performance": self.monthly,
            "loan_outcomes": self.outcomes,
        }


def amortized_payment(principal: np.ndarray, apr: np.ndarray, term: np.ndarray) -> np.ndarray:
    """Level monthly payment for a fully amortizing simple-interest loan."""
    r = np.asarray(apr, dtype=float) / 12.0
    principal = np.asarray(principal, dtype=float)
    term = np.asarray(term, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = principal * r / (1.0 - (1.0 + r) ** (-term))
    return np.where(r == 0, principal / term, pay)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _month_index(month: str, first: pd.Period) -> int:
    return (pd.Period(month, freq="M") - first).n


# ----------------------------------------------------------------------
# Origination
# ----------------------------------------------------------------------
def _originate(cfg: SimConfig, rng: np.random.Generator, months: pd.PeriodIndex) -> dict[str, np.ndarray]:
    n = cfg.n_loans
    first = months[0]
    n_orig_months = _month_index(cfg.last_origination, first) + 1

    # Origination month: volume grows slowly and is a bit higher in spring.
    m = np.arange(n_orig_months)
    month_of_year = np.array([months[i].month for i in m])
    weights = (1.0 + 0.004 * m) * (1.0 + 0.12 * np.isin(month_of_year, [3, 4, 5]))
    orig_t = rng.choice(m, size=n, p=weights / weights.sum())

    # Borrowers. A small share of loans belongs to a borrower who has another loan.
    n_borrowers = max(1, int(round(n * (1.0 - cfg.repeat_borrower_share))))
    borrower = np.concatenate([np.arange(n_borrowers), rng.integers(0, n_borrowers, n - n_borrowers)])
    rng.shuffle(borrower)

    # Hidden borrower traits. These are never written to the output tables.
    z_b = rng.normal(size=n_borrowers)
    habitual_late_b = rng.random(n_borrowers) < cfg.habitual_late_share
    irregular_b = rng.random(n_borrowers) < cfg.irregular_income_share
    income_noise_b = rng.normal(size=n_borrowers)
    score_noise_b = rng.normal(size=n_borrowers)

    z = z_b[borrower].copy()
    loose = (orig_t >= _month_index(cfg.loose_vintage_start, first)) & (
        orig_t <= _month_index(cfg.loose_vintage_end, first)
    )
    z = z - cfg.loose_vintage_shift * loose  # looser underwriting lets in weaker borrowers

    rho = cfg.score_latent_corr
    score_std = rho * z + np.sqrt(1 - rho**2) * score_noise_b[borrower] + 0.25 * rng.normal(size=n)
    credit_score = np.clip(np.round(cfg.score_mean + cfg.score_sd * score_std), 450, 820).astype(int)
    income = np.exp(cfg.income_log_mean + 0.12 * score_std + cfg.income_log_sd * income_noise_b[borrower])
    income = np.clip(income, 1_600, 16_000)

    # Vehicle
    used = rng.random(n) < cfg.used_share
    vehicle_age = np.where(used, np.clip(rng.geometric(0.22, n), 1, 14), 0)
    base_value = np.where(used, 31_000 * np.exp(-0.105 * vehicle_age), 34_000)
    vehicle_value = np.clip(base_value * rng.lognormal(0.0, 0.28, n), 4_000, 75_000)
    mileage = np.where(used, vehicle_age * rng.normal(12_500, 2_500, n), rng.uniform(5, 200, n))
    mileage = np.clip(mileage, 5, 220_000)

    # Deal structure: price, taxes and fees, down payment, amount financed.
    price = vehicle_value * (1.0 + rng.uniform(0.0, 0.10, n))
    taxes_fees = 0.08 * price + 600
    dp_ratio = np.clip(rng.beta(2.0, 9.0, n) * 0.9 + 0.0004 * (credit_score - 600), 0.0, 0.45)
    down_payment = dp_ratio * price
    loan_amount = price + taxes_fees - down_payment
    ltv = loan_amount / vehicle_value

    terms = np.array([36, 48, 60, 66, 72, 75])
    size_rank = np.clip((loan_amount - 8_000) / 30_000, 0, 1)
    term_probs = np.stack(
        [
            0.10 * (1 - size_rank),
            0.20 * (1 - size_rank) + 0.03,
            0.30 - 0.05 * size_rank,
            0.12 + 0.02 * size_rank,
            0.20 + 0.22 * size_rank,
            0.05 + 0.11 * size_rank,
        ],
        axis=1,
    )
    term_probs /= term_probs.sum(axis=1, keepdims=True)
    u = rng.random(n)
    term = terms[(u[:, None] > np.cumsum(term_probs, axis=1)).sum(axis=1).clip(max=5)]

    apr = (
        0.205
        - 0.0005 * (credit_score - 550)
        + 0.03 * (ltv - 1.0)
        + 0.0002 * (term - 60)
        + 0.01 * used
        + rng.normal(0, 0.012, n)
    )
    apr = np.round(np.clip(apr, 0.035, 0.2999), 4)
    payment = np.round(amortized_payment(loan_amount, apr, term), 2)

    # Payment-to-income cap. Applications over the cap would not be approved,
    # so the simulated income is raised to the minimum that passes.
    cap = cfg.max_pti * rng.uniform(0.82, 1.0, n)
    income = np.maximum(income, payment / cap)
    pti = payment / income

    groups = np.array(["A", "B", "C"])
    return {
        "orig_t": orig_t,
        "borrower": borrower,
        "z": z,
        "habitual_late": habitual_late_b[borrower],
        "irregular": irregular_b[borrower],
        "skip": rng.random(n) < cfg.skip_share,
        "dep_rate": np.clip(rng.normal(0.015, 0.003, n), 0.006, 0.03),
        "credit_score": credit_score,
        "income": np.round(income, 0),
        "used": used,
        "vehicle_age": vehicle_age,
        "vehicle_value": np.round(vehicle_value, 0),
        "mileage": np.round(mileage, 0),
        "down_payment": np.round(down_payment, 2),
        "loan_amount": np.round(loan_amount, 2),
        "ltv": ltv,
        "term": term,
        "apr": apr,
        "payment": payment,
        "pti": pti,
        "due_day": rng.integers(1, 26, n),
        "servicing_group": groups[rng.choice(3, size=n, p=cfg.servicing_group_shares)],
        "channel": np.where(rng.random(n) < 0.55, "franchise", "independent"),
    }


# ----------------------------------------------------------------------
# Monthly simulation
# ----------------------------------------------------------------------
def simulate_portfolio(cfg: SimConfig | None = None) -> Portfolio:
    """Run the simulator and return the linked tables."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    first = pd.Period(cfg.first_origination, freq="M")
    months = pd.period_range(first, pd.Period(cfg.data_cutoff, freq="M"), freq="M")
    n_months = len(months)
    n = cfg.n_loans
    o = _originate(cfg, rng, months)

    # Calendar-level series: macro stress and repossession capacity.
    macro = np.zeros(n_months)
    for t in range(1, n_months):
        macro[t] = cfg.macro_ar * macro[t - 1] + rng.normal(0, cfg.macro_sd)
    s0, s1 = _month_index(cfg.macro_shock_start, first), _month_index(cfg.macro_shock_end, first)
    macro[max(s0, 0) : max(s1 + 1, 0)] += cfg.macro_shock_size
    capacity = rng.lognormal(0.0, cfg.repo_capacity_sd, n_months)
    p0, p1 = _month_index(cfg.repo_pause_start, first), _month_index(cfg.repo_pause_end, first)
    capacity[max(p0, 0) : max(p1 + 1, 0)] *= cfg.repo_pause_factor
    tax_season = np.array([p.month in (2, 3, 4) for p in months])

    group_idx = np.searchsorted(np.array(["A", "B", "C"]), o["servicing_group"])
    assign_table = np.array([cfg.repo_assign_probs[g] for g in ("A", "B", "C")])

    # Loan state
    active = np.zeros(n, dtype=bool)
    mob = np.zeros(n, dtype=int)
    bal = o["loan_amount"].astype(float).copy()
    behind = np.zeros(n)  # payments behind, in units of the scheduled payment
    n_unpaid = np.zeros(n, dtype=int)  # unpaid installments at the last month end
    dpd = np.zeros(n, dtype=int)
    stressed = np.zeros(n, dtype=bool)
    assigned = np.zeros(n, dtype=bool)
    n_ext = np.zeros(n, dtype=int)
    final_status = np.full(n, -1)
    close_t = np.full(n, -1)
    close_day = np.zeros(n, dtype=int)
    default_t = np.full(n, -1)
    assign_t = np.full(n, -1)
    voluntary = np.zeros(n, dtype=bool)
    recovery_amt = np.full(n, np.nan)
    close_balance = np.full(n, np.nan)

    payment, apr, z = o["payment"], o["apr"], o["z"]
    rec: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "loan", "t", "mob", "sched", "paid", "pay_day", "beg_bal", "end_bal",
            "behind", "dpd", "status", "extension", "assigned", "veh_value",
        )
    }

    def _write(idx, t, sched, paid, pay_day, beg, end, status, ext):
        rec["loan"].append(idx)
        rec["t"].append(np.full(idx.size, t))
        rec["mob"].append(mob[idx].copy())
        rec["sched"].append(sched)
        rec["paid"].append(paid)
        rec["pay_day"].append(pay_day)
        rec["beg_bal"].append(beg)
        rec["end_bal"].append(end)
        rec["behind"].append(behind[idx].copy())
        rec["dpd"].append(dpd[idx].copy())
        rec["status"].append(status)
        rec["extension"].append(ext)
        rec["assigned"].append(assigned[idx].copy())
        rec["veh_value"].append(o["vehicle_value"][idx] * np.exp(-o["dep_rate"][idx] * mob[idx]))

    for t in range(n_months):
        active |= o["orig_t"] == t - 1  # first payment is due the month after origination
        act = np.flatnonzero(active)
        if act.size == 0:
            continue
        mob[act] += 1
        cur_ltv = bal[act] / (o["vehicle_value"][act] * np.exp(-o["dep_rate"][act] * mob[act]))

        # --- servicing actions based on last month-end delinquency -------
        nu = n_unpaid[act]
        assigned[act[nu < 2]] = False  # borrower cured, assignment cancelled
        u = rng.random(act.size)
        recovered = assigned[act] & ~o["skip"][act] & (u < cfg.repo_recovery_prob)
        surrender = stressed[act] & (nu >= 2) & (rng.random(act.size) < cfg.voluntary_surrender_prob)
        repo_now = recovered | surrender
        charge_now = (nu >= 6) & ~repo_now & (rng.random(act.size) < cfg.chargeoff_prob)

        bucket = np.clip(nu - 3, 0, 2)
        p_assign = np.where(nu >= 3, assign_table[group_idx[act], bucket] * capacity[t], 0.0)
        new_assign = ~assigned[act] & ~repo_now & ~charge_now & (rng.random(act.size) < p_assign)
        assigned[act[new_assign]] = True
        assign_t[act[new_assign & (assign_t[act] < 0)]] = t

        term_mask = repo_now | charge_now
        if term_mask.any():
            idx = act[term_mask]
            status = np.where(repo_now[term_mask], REPOSSESSED, CHARGED_OFF)
            final_status[idx] = status
            close_t[idx] = t
            close_day[idx] = rng.integers(1, 29, idx.size)
            voluntary[idx] = surrender[term_mask] & ~recovered[term_mask]
            close_balance[idx] = bal[idx]
            value_now = o["vehicle_value"][idx] * np.exp(-o["dep_rate"][idx] * mob[idx])
            proceeds = np.minimum(value_now * rng.uniform(0.45, 0.80, idx.size) - 900, bal[idx]).clip(min=0)
            recovery_amt[idx] = np.where(status == REPOSSESSED, proceeds, 0.0)
            default_t[idx] = np.where(default_t[idx] < 0, t, default_t[idx])
            dpd[idx] += close_day[idx]
            zeros = np.zeros(idx.size)
            _write(idx, t, zeros, zeros, np.zeros(idx.size, dtype=int), bal[idx].copy(), bal[idx].copy(),
                   status, np.zeros(idx.size, dtype=bool))
            active[idx] = False
            assigned[idx] = False
            act, cur_ltv = act[~term_mask], cur_ltv[~term_mask]
            if act.size == 0:
                continue

        # --- hidden stress regime ---------------------------------------
        k = act.size
        enter_logit = (
            cfg.stress_base_logit
            - cfg.stress_z_coef * z[act]
            + cfg.stress_pti_coef * (o["pti"][act] - 0.13) / 0.04
            + cfg.stress_ltv_coef * (o["ltv"][act] - 1.0) / 0.15
            + cfg.stress_early_life_bump * (mob[act] <= 6)
            + macro[t]
        )
        p_exit = cfg.stress_exit_prob * np.where(behind[act] >= 3, 0.5, 1.0) * (1.3 if tax_season[t] else 1.0)
        u = rng.random(k)
        stressed[act] = np.where(stressed[act], u >= p_exit, u < _sigmoid(enter_logit))
        st = stressed[act]

        # --- payment behavior -------------------------------------------
        b = behind[act]
        is_behind = b > PARTIAL_TOLERANCE
        risk = _sigmoid(-z[act])
        p_miss = np.where(
            st,
            np.clip(0.40 + 0.06 * np.minimum(b, 4) + 0.10 * np.clip(cur_ltv - 1.0, 0, 0.5), 0, 0.80),
            np.where(is_behind, 0.10, 0.006 + 0.012 * risk),
        )
        p_partial = np.where(st, 0.20, np.where(is_behind, 0.08, 0.012))
        p_extra = np.where(st, 0.02, np.where(is_behind, 0.30 + (0.15 if tax_season[t] else 0.0), 0.0))
        u = rng.random(k)
        miss = u < p_miss
        partial = ~miss & (u < p_miss + p_partial)
        extra = ~miss & ~partial & (u < p_miss + p_partial + p_extra)

        interest = bal[act] * apr[act] / 12.0
        payoff_amt = bal[act] + interest
        sched = np.minimum(payment[act], payoff_amt)

        mult = np.ones(k)
        mult[miss] = 0.0
        mult[partial] = rng.uniform(0.30, 0.85, partial.sum())
        full = ~miss & ~partial & ~extra
        irr = full & o["irregular"][act]
        mult[irr] = rng.uniform(0.80, 1.36, irr.sum())
        paid = mult * sched
        # catching up: either everything owed, or one extra installment
        pay_all = rng.random(k) < 0.6
        paid = np.where(extra, sched + payment[act] * np.where(pay_all, b, np.minimum(b, 1.0)), paid)

        # voluntary prepayment (trade-in, refinance, cash payoff)
        p_prepay = (cfg.prepay_base + cfg.prepay_age_slope * np.minimum(mob[act], 36)) * (
            1.0 + 0.004 * (o["credit_score"][act] - 600)
        ).clip(0.3, 2.0)
        prepay = ~is_behind & ~st & (rng.random(k) < p_prepay)
        paid = np.where(prepay, payoff_amt, paid)
        paid = np.round(np.minimum(paid, payoff_amt), 2)

        # payment timing inside the month
        due = o["due_day"][act]
        p_late = np.where(o["habitual_late"][act], 0.50, 0.04 + 0.05 * risk)
        late = (rng.random(k) < p_late) | is_behind | st
        on_time_day = np.maximum(due - rng.integers(0, 6, k), 1)
        late_day = due + np.minimum(rng.integers(1, 21, k), 30 - due)
        pay_day = np.where(paid > 0, np.where(late, late_day, on_time_day), 0)

        beg = bal[act].copy()
        end = np.maximum(payoff_amt - paid, 0.0)
        end[end < 1.0] = 0.0
        bal[act] = end
        behind[act] = np.maximum(b + (sched - paid) / payment[act], 0.0)

        # extension: the servicer moves past-due installments to the end of the loan
        nu_now = np.ceil(behind[act] - PARTIAL_TOLERANCE).clip(min=0).astype(int)
        ext = (
            (nu_now >= 2) & (nu_now <= 3) & (paid > 0) & (n_ext[act] < cfg.max_extensions)
            & (rng.random(k) < cfg.extension_prob)
        )
        behind[act[ext]] = np.maximum(behind[act[ext]] - 2.0, 0.0)
        n_ext[act[ext]] += 1

        paid_off = end == 0.0
        behind[act[paid_off]] = 0.0
        nu_now = np.ceil(behind[act] - PARTIAL_TOLERANCE).clip(min=0).astype(int)
        n_unpaid[act] = nu_now
        dpd[act] = np.where(nu_now > 0, (nu_now - 1) * 30 + (30 - due), 0)
        new_default = (nu_now >= 5) & (default_t[act] < 0)
        default_t[act[new_default]] = t

        status = np.select(
            [paid_off, dpd[act] == 0, dpd[act] < 30, dpd[act] < 60, dpd[act] < 90],
            [PAID_OFF, 0, 1, 2, 3],
            default=4,
        )
        _write(act, t, np.round(sched, 2), paid, pay_day, beg, end, status, ext)

        done = act[paid_off]
        final_status[done] = PAID_OFF
        close_t[done] = t
        close_day[done] = pay_day[paid_off]
        close_balance[done] = 0.0
        active[done] = False
        assigned[done] = False

    return _assemble(cfg, o, rec, months, final_status, close_t, close_day, default_t, assign_t,
                     voluntary, recovery_amt, close_balance, n_ext)


# ----------------------------------------------------------------------
# Output tables
# ----------------------------------------------------------------------
def _dates(months: pd.PeriodIndex, t: np.ndarray, day: np.ndarray) -> pd.Series:
    """Calendar date for month index ``t`` and day of month ``day`` (0 -> missing)."""
    start = months.to_timestamp(how="start")
    dim = months.days_in_month.to_numpy()
    valid = (t >= 0) & (day > 0)
    tt = np.where(valid, t, 0)
    d = np.minimum(np.where(valid, day, 1), dim[tt])
    out = pd.Series(start[tt] + pd.to_timedelta(d - 1, unit="D"))
    out[~valid] = pd.NaT
    return out


def _assemble(cfg, o, rec, months, final_status, close_t, close_day, default_t, assign_t,
              voluntary, recovery_amt, close_balance, n_ext) -> Portfolio:
    n = cfg.n_loans
    loan_ids = np.array([f"L{i:07d}" for i in range(n)])
    borrower_ids = np.array([f"B{i:07d}" for i in o["borrower"]])
    month_end = months.to_timestamp(how="end").normalize()

    loans = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "borrower_id": borrower_ids,
            "origination_date": _dates(months, o["orig_t"], np.minimum(o["due_day"] + 2, 28)),
            "channel": o["channel"],
            "servicing_group": o["servicing_group"],
            "vehicle_type": np.where(o["used"], "used", "new"),
            "vehicle_age_years": o["vehicle_age"],
            "vehicle_mileage": o["mileage"].astype(int),
            "vehicle_value": o["vehicle_value"],
            "loan_amount": o["loan_amount"],
            "down_payment": o["down_payment"],
            "ltv": np.round(o["ltv"], 4),
            "apr": o["apr"],
            "term_months": o["term"],
            "scheduled_payment": o["payment"],
            "payment_due_day": o["due_day"],
            "credit_score": o["credit_score"],
            "monthly_income": o["income"],
            "pti": np.round(o["pti"], 4),
        }
    )

    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in rec.items()}
    li, t = cat["loan"].astype(int), cat["t"].astype(int)
    status = cat["status"].astype(int)
    monthly = pd.DataFrame(
        {
            "loan_id": loan_ids[li],
            "as_of_date": month_end[t],
            "months_on_book": cat["mob"].astype(int),
            "beginning_balance": np.round(cat["beg_bal"], 2),
            "ending_balance": np.round(cat["end_bal"], 2),
            "scheduled_amount": cat["sched"],
            "amount_paid": cat["paid"],
            "payments_behind": np.round(cat["behind"], 3),
            "amount_past_due": np.round(cat["behind"] * o["payment"][li], 2),
            "days_past_due": cat["dpd"].astype(int),
            "status": pd.Categorical.from_codes(status, STATUS_NAMES),
            "extension_granted": cat["extension"].astype(bool),
            "repo_assigned": cat["assigned"].astype(bool),
            "est_vehicle_value": np.round(cat["veh_value"], 0),
        }
    )

    is_active_row = status < PAID_OFF
    has_due = is_active_row | (status == PAID_OFF)
    pay_day = cat["pay_day"].astype(int)
    due_day = o["due_day"][li]
    payments = pd.DataFrame(
        {
            "loan_id": loan_ids[li],
            "due_date": _dates(months, t, due_day),
            "scheduled_amount": cat["sched"],
            "amount_paid": cat["paid"],
            "payment_date": _dates(months, t, pay_day),
            "days_late": np.where(pay_day > 0, np.maximum(pay_day - due_day, 0), -1),
        }
    )[has_due].reset_index(drop=True)
    payments["days_late"] = payments["days_late"].where(payments["days_late"] >= 0).astype("Int64")

    closed = final_status >= 0
    outcomes = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "final_status": np.where(closed, np.array(STATUS_NAMES)[np.clip(final_status, 0, 7)], "active"),
            "censored": ~closed,  # still open at the data cutoff
            "close_date": _dates(months, close_t, close_day),
            "default_date": pd.Series(month_end[np.clip(default_t, 0, None)]).where(default_t >= 0),
            "repo_assigned_date": pd.Series(month_end[np.clip(assign_t, 0, None)]).where(assign_t >= 0),
            "repossession_date": _dates(months, np.where(final_status == REPOSSESSED, close_t, -1), close_day),
            "voluntary_surrender": voluntary,
            "balance_at_close": np.round(close_balance, 2),
            "recovery_amount": np.round(recovery_amt, 2),
            "extensions_granted": n_ext,
        }
    )
    # a repossession or charge-off in the same month end as the default flag keeps the earlier date
    outcomes["default_date"] = outcomes[["default_date", "close_date"]].min(axis=1).where(default_t >= 0)

    monthly = monthly.sort_values(["loan_id", "as_of_date"], kind="stable").reset_index(drop=True)
    payments = payments.sort_values(["loan_id", "due_date"], kind="stable").reset_index(drop=True)
    logger.info("Simulated %d loans, %d loan-months", n, len(monthly))
    return Portfolio(loans=loans, payments=payments, monthly=monthly, outcomes=outcomes, config=cfg)
