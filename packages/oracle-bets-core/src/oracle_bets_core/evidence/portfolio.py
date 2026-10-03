"""
Offline Kelly research; no execution or evidence-store writes.

Odds and realized returns must already include fees and executable-price costs.
Inputs are a preregistered opportunity population with probabilities frozen at
``decided_at``. This module cannot establish that provenance from floats alone.
Absent supplied joint scenarios, marginal Kelly sizes are conservatively capped
by fixture and total open exposure: they are NOT an independence-based optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import groupby

import numpy as np
from scipy.optimize import minimize

from oracle_bets_core.betting import kelly_fraction

_MIN_FIXTURES = 20
_MIN_BLOCKS = 8
_MIN_SAMPLES = 100
_NUMERIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class KellyPolicy:
    name: str = "balanced_quarter"
    fraction: float = 0.25
    ticket_cap: float = 0.02
    fixture_cap: float = 0.05
    total_cap: float = 0.20

    def __post_init__(self) -> None:
        if not self.name.strip() or not 0 < self.fraction <= 1:
            raise ValueError("policy requires a name and Kelly fraction in (0, 1]")
        if not 0 < self.ticket_cap <= self.fixture_cap <= self.total_cap < 1:
            raise ValueError("caps must satisfy 0 < ticket <= fixture <= total < 1")


@dataclass(frozen=True)
class RiskTarget:
    """Research preferences, not a guaranteed drawdown limit or optimal sizing."""

    drawdown_limit: float = 0.20
    drawdown_penalty: float = 1.0
    ruin_floor: float = 0.10
    min_fixtures: int = 50
    min_blocks: int = 8
    bootstrap_samples: int = 1000
    seed: int = 7

    def __post_init__(self) -> None:
        if not 0 < self.drawdown_limit < 1 or not 0 < self.ruin_floor < 1:
            raise ValueError("drawdown limit and ruin floor must be in (0, 1)")
        if not math.isfinite(self.drawdown_penalty) or self.drawdown_penalty < 0:
            raise ValueError("drawdown penalty must be finite and non-negative")
        if (
            self.min_fixtures < _MIN_FIXTURES
            or self.min_blocks < _MIN_BLOCKS
            or self.bootstrap_samples < _MIN_SAMPLES
        ):
            raise ValueError(
                "research requires at least 20 fixtures, 8 blocks, 100 samples"
            )


BALANCED_TARGET = RiskTarget()

BALANCED_POLICIES = (
    KellyPolicy("balanced_eighth", 0.125, 0.01, 0.025, 0.10),
    KellyPolicy(),
    KellyPolicy("balanced_half", 0.5, 0.025, 0.075, 0.25),
)


@dataclass(frozen=True)
class Opportunity:
    ticket_id: str
    fixture_id: str
    decided_at: datetime
    settled_at: datetime | None
    probability: float  # P(win | non-push/non-void); scenarios support richer payouts
    decimal_odds: float
    realized_return: float | None  # net profit / stake; None for unsettled quotes

    def __post_init__(self) -> None:
        if not self.ticket_id.strip() or not self.fixture_id.strip():
            raise ValueError("ticket and stable sporting fixture IDs are required")
        if self.decided_at.utcoffset() is None or (
            self.settled_at is not None and self.settled_at.utcoffset() is None
        ):
            raise ValueError("decision and settlement times must be timezone aware")
        if (self.settled_at is None) != (self.realized_return is None):
            raise ValueError(
                "settlement time and realized return must both be known or missing"
            )
        if self.settled_at is not None and self.settled_at <= self.decided_at:
            raise ValueError("settlement must follow decision")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite in [0, 1]")
        if not math.isfinite(self.decimal_odds) or self.decimal_odds <= 1:
            raise ValueError("net decimal odds must be finite and greater than one")
        if self.realized_return is not None and (
            not math.isfinite(self.realized_return)
            or not -1 <= self.realized_return <= self.decimal_odds - 1
        ):
            raise ValueError(
                "realized net return must be between -1 and odds minus one"
            )


@dataclass(frozen=True)
class JointScenarios:
    """
    Joint, ex-ante net payoffs; columns match tickets, rows are joint worlds.

    Probabilities are authoritative for scenario allocation, replacing marginal
    binary probabilities. They must come from a validated joint model, never a
    product of marginal probabilities for correlated map/prop/handicap tickets.
    """

    ticket_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    returns: tuple[tuple[float, ...], ...]
    available_at: datetime


def allocate_portfolio(
    opportunities: tuple[Opportunity, ...],
    policy: KellyPolicy,
    *,
    equity: float,
    open_stakes: dict[str, float] | None = None,
    scenarios: JointScenarios | None = None,
) -> dict[str, float]:
    """
    Size one simultaneous decision batch; reserve all unsettled stakes.

    Joint allocation maximizes expected log wealth under shared constraints, then
    applies the Kelly fraction. Existing positions are conservatively assumed
    lost in this optimization; no unsupported joint distribution is fabricated.
    Without scenarios, binary marginal sizes are scaled down proportionally to
    fit remaining fixture/global budgets. No hedge credits are granted.
    """
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")
    ids = [row.ticket_id for row in opportunities]
    if len(set(ids)) != len(ids) or len({r.decided_at for r in opportunities}) > 1:
        raise ValueError("allocation requires unique tickets at one decision time")
    existing = dict(open_stakes or {})
    if any(not math.isfinite(value) or value < 0 for value in existing.values()):
        raise ValueError("open stakes must be finite and non-negative")
    reserved = sum(existing.values())
    if reserved > equity + 1e-9:
        raise ValueError("open stakes cannot exceed equity")
    budget = max(0.0, min(equity - reserved, equity * policy.total_cap - reserved))
    groups = {
        fixture: [i for i, row in enumerate(opportunities) if row.fixture_id == fixture]
        for fixture in {row.fixture_id for row in opportunities}
    }
    limits = {
        fixture: max(0.0, equity * policy.fixture_cap - existing.get(fixture, 0.0))
        for fixture in groups
    }
    if scenarios is not None:
        stakes = _scenario_stakes(
            opportunities, policy, equity, reserved, budget, groups, limits, scenarios
        )
    else:
        stakes = np.array(
            [
                min(
                    equity * policy.ticket_cap,
                    equity
                    * kelly_fraction(
                        row.decimal_odds, row.probability, fraction=policy.fraction
                    ),
                )
                for row in opportunities
            ]
        )
        for fixture, indices in groups.items():
            total = float(stakes[indices].sum())
            if total > limits[fixture]:
                stakes[indices] *= limits[fixture] / total
        total = float(stakes.sum())
        if total > budget:
            stakes *= budget / total
    return dict(zip(ids, map(float, stakes), strict=True))


def _scenario_stakes(rows, policy, equity, reserved, budget, groups, limits, scenarios):
    if scenarios.ticket_ids != tuple(row.ticket_id for row in rows) or not rows:
        raise ValueError(
            "joint scenario columns must match the decision tickets exactly"
        )
    if (
        scenarios.available_at.utcoffset() is None
        or scenarios.available_at > rows[0].decided_at
    ):
        raise ValueError("joint scenarios must be available at decision time")
    probability = np.asarray(scenarios.probabilities, dtype=float)
    returns = np.asarray(scenarios.returns, dtype=float)
    if (
        probability.ndim != 1
        or not probability.size
        or returns.shape != (probability.size, len(rows))
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0)
        or not np.isclose(probability.sum(), 1.0, atol=1e-10, rtol=0)
        or not np.all(np.isfinite(returns))
        or np.any(returns < -1)
        or np.any(returns > np.array([row.decimal_odds - 1 for row in rows]))
    ):
        raise ValueError(
            "joint scenarios require normalized probabilities and valid net returns"
        )
    # Full-Kelly optimization, with caps scaled so fractional stakes respect them.
    wealth = 1 - reserved / equity
    total_limit = max(0.0, min(budget / equity / policy.fraction, wealth - 1e-10))
    constraints = [np.ones(len(rows))]
    caps = [total_limit]
    for fixture, indices in groups.items():
        mask = np.zeros(len(rows))
        mask[indices] = 1.0
        constraints.append(mask)
        caps.append(limits[fixture] / equity / policy.fraction)
    matrix, cap_vector = np.asarray(constraints), np.asarray(caps)

    def objective(amounts):
        return -float(probability @ np.log(wealth + returns @ amounts))

    def gradient(amounts):
        return -(probability / (wealth + returns @ amounts)) @ returns

    if budget <= 0:
        return np.zeros(len(rows))
    result = minimize(
        objective,
        np.zeros(len(rows)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, policy.ticket_cap / policy.fraction)] * len(rows),
        constraints={
            "type": "ineq",
            "fun": lambda x: cap_vector - matrix @ x,
            "jac": lambda _x: -matrix,
        },
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not result.success or np.any(
        matrix @ result.x > cap_vector + _NUMERIC_TOLERANCE
    ):
        raise ValueError(
            "joint allocation failed to converge within exposure constraints"
        )
    return np.maximum(0.0, result.x) * equity * policy.fraction


@dataclass(frozen=True)
class BootstrapRisk:
    """Empirical block-resampling stress results, not population guarantees."""

    log_growth_p05: float
    log_growth_p95: float
    drawdown_p95: float
    ruin_frequency: float
    maximum_open_exposure_p95: float
    samples: int
    seed: int


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: KellyPolicy
    fixture_count: int
    staked_fixture_count: int
    block_count: int
    final_equity: float
    log_growth: float
    maximum_drawdown: float
    maximum_open_exposure: float
    ruined: bool
    stakes: dict[str, float]
    bootstrap: BootstrapRisk | None


def _ordered(rows):
    ordered = tuple(sorted(rows, key=lambda row: (row.decided_at, row.ticket_id)))
    if any(row.settled_at is None for row in ordered):
        raise ValueError(
            "research replay requires settled outcomes for every opportunity"
        )
    if len({row.ticket_id for row in ordered}) != len(ordered):
        raise ValueError("ticket IDs must be unique")
    return ordered


def _closed_blocks(rows):
    """Weekly-or-longer blocks; merge until no fixture or settlement crosses."""
    if not rows:
        return []
    fixture_end = {}
    for row in rows:
        fixture_end[row.fixture_id] = max(
            fixture_end.get(row.fixture_id, row.settled_at), row.settled_at
        )
    blocks = []
    current = []
    end = rows[0].decided_at + timedelta(days=7)
    for row in rows:
        if current and row.decided_at >= end:
            blocks.append(tuple(current))
            current = []
            end = row.decided_at + timedelta(days=7)
        current.append(row)
        end = max(end, fixture_end[row.fixture_id])
    blocks.append(tuple(current))
    return blocks


def _simulate(rows, policy, initial_equity, ruin_floor):
    equity = peak = initial_equity
    drawdown = exposure = 0.0
    ruined = False
    active = []
    stakes = {}
    path = [1.0]

    def settle(until):
        nonlocal equity, peak, drawdown, exposure, ruined, active
        due = sorted(
            (item for item in active if until is None or item[0].settled_at <= until),
            key=lambda item: item[0].settled_at,
        )
        for _, group in groupby(due, key=lambda item: item[0].settled_at):
            batch = tuple(group)
            equity += sum(stake * row.realized_return for row, stake in batch)
            completed = {row.ticket_id for row, _ in batch}
            active = [
                (row, stake) for row, stake in active if row.ticket_id not in completed
            ]
            peak = max(peak, equity)
            drawdown = max(drawdown, 1 - equity / peak)
            exposure = max(exposure, sum(stake for _, stake in active) / equity)
            ruined |= equity <= initial_equity * ruin_floor
            path.append(equity / initial_equity)

    for decision, group in groupby(rows, key=lambda row: row.decided_at):
        settle(decision)
        batch = tuple(group)
        opened = {}
        for row, stake in active:
            opened[row.fixture_id] = opened.get(row.fixture_id, 0.0) + stake
        allocated = allocate_portfolio(batch, policy, equity=equity, open_stakes=opened)
        stakes.update(allocated)
        active.extend(
            (row, allocated[row.ticket_id])
            for row in batch
            if allocated[row.ticket_id] > 0
        )
        exposure = max(exposure, sum(stake for _, stake in active) / equity)
    settle(None)
    return equity, drawdown, exposure, ruined, stakes, path


def evaluate_policy(
    opportunities: tuple[Opportunity, ...],
    policy: KellyPolicy,
    *,
    initial_equity: float = 100.0,
    target: RiskTarget = BALANCED_TARGET,
) -> PolicyEvaluation:
    """
    Replay one locked capped policy without crediting unsettled winnings.

    Drawdown uses equity with open positions held at cost, not mark-to-market.
    Resample complete closed exposure blocks, preserving simultaneous tickets,
    fixture dependence, holding times, and within-block settlement paths. Long
    overlap reduces the number of usable blocks instead of inventing independence.
    """
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial equity must be finite and positive")
    rows = _ordered(opportunities)
    final, dd, exposure, ruined, stakes, _ = _simulate(
        rows, policy, initial_equity, target.ruin_floor
    )
    staked_fixtures = {row.fixture_id for row in rows if stakes[row.ticket_id] > 0}
    blocks = _closed_blocks(rows)
    risk = None
    if len(staked_fixtures) >= target.min_fixtures and len(blocks) >= target.min_blocks:
        simulations = [
            _simulate(block, policy, 1.0, target.ruin_floor) for block in blocks
        ]
        paths = [simulation[-1] for simulation in simulations]
        rng = np.random.default_rng(target.seed)
        growth, drawdowns, ruins, exposures = [], [], [], []
        for _ in range(target.bootstrap_samples):
            equity = peak = 1.0
            maximum = 0.0
            hit_ruin = False
            maximum_exposure = 0.0
            for index in rng.integers(0, len(paths), len(paths)):
                maximum_exposure = max(maximum_exposure, simulations[index][2])
                scaled = equity * np.asarray(paths[index])
                peaks = np.maximum.accumulate(np.concatenate(([peak], scaled)))[1:]
                maximum = max(maximum, float(np.max(1 - scaled / peaks)))
                hit_ruin |= bool(np.any(scaled <= target.ruin_floor))
                equity = float(scaled[-1])
                peak = float(peaks[-1])
            growth.append(math.log(equity))
            drawdowns.append(maximum)
            ruins.append(hit_ruin)
            exposures.append(maximum_exposure)
        risk = BootstrapRisk(
            float(np.quantile(growth, 0.05)),
            float(np.quantile(growth, 0.95)),
            float(np.quantile(drawdowns, 0.95)),
            float(np.mean(ruins)),
            float(np.quantile(exposures, 0.95)),
            target.bootstrap_samples,
            target.seed,
        )
    return PolicyEvaluation(
        policy,
        len({row.fixture_id for row in rows}),
        len(staked_fixtures),
        len(blocks),
        final,
        math.log(final / initial_equity),
        dd,
        exposure,
        ruined,
        stakes,
        risk,
    )


@dataclass(frozen=True)
class ResearchReport:
    status: str
    reason: str
    target: RiskTarget
    selected_policy: KellyPolicy | None = None
    development: tuple[PolicyEvaluation, ...] = ()
    holdout: PolicyEvaluation | None = None
    assumptions: tuple[str, ...] = field(
        default=(
            "Inputs require frozen ex-ante probabilities, net executable prices, and complete opportunities.",
            "Unknown dependence uses conservative fixture/global caps; no independence claim.",
            "Drawdown holds open tickets at cost; block bootstrap is empirical stress, not a risk guarantee.",
            "Policy selection uses development only; the heldout result cannot be used to retune it.",
        )
    )


def select_policy(
    development: tuple[Opportunity, ...],
    holdout: tuple[Opportunity, ...],
    *,
    candidates: tuple[KellyPolicy, ...] = BALANCED_POLICIES,
    target: RiskTarget = BALANCED_TARGET,
) -> ResearchReport:
    """
    Select development log-growth minus drawdown penalty, then lock and test.

    A policy must also pass the development block-stress drawdown limit. A
    nonpositive objective selects no strategy. This is research evidence, never
    automatic activation or a claim that the winning candidate is universally best.
    """
    if not candidates or len({policy.name for policy in candidates}) != len(candidates):
        raise ValueError("candidate policy names must be unique and non-empty")
    dev, test = _ordered(development), _ordered(holdout)
    if not dev or not test:
        return ResearchReport(
            "unavailable",
            "Separate development and heldout evidence is required.",
            target,
        )
    if {r.fixture_id for r in dev} & {r.fixture_id for r in test}:
        raise ValueError(
            "a sporting fixture cannot occur in both development and holdout"
        )
    if {r.ticket_id for r in dev} & {r.ticket_id for r in test}:
        raise ValueError("a ticket cannot occur in both development and holdout")
    if max(row.settled_at for row in dev) > min(row.decided_at for row in test):
        raise ValueError(
            "all development outcomes must be settled before heldout decisions"
        )
    evaluations = tuple(
        evaluate_policy(dev, policy, target=target) for policy in candidates
    )
    eligible = [result for result in evaluations if result.bootstrap is not None]
    if not eligible:
        return ResearchReport(
            "insufficient_evidence",
            "Too few staked fixtures or closed temporal blocks.",
            target,
            development=evaluations,
        )
    eligible = [
        result
        for result in eligible
        if result.bootstrap is not None
        and result.bootstrap.drawdown_p95 <= target.drawdown_limit
        and result.maximum_drawdown <= target.drawdown_limit
        and result.bootstrap.ruin_frequency == 0
        and result.log_growth - target.drawdown_penalty * result.maximum_drawdown > 0
    ]
    if not eligible:
        return ResearchReport(
            "no_eligible_policy",
            "No candidate passes development growth and risk criteria.",
            target,
            development=evaluations,
        )
    chosen = max(
        eligible,
        key=lambda result: (
            result.log_growth - target.drawdown_penalty * result.maximum_drawdown,
            -result.policy.fraction,
            result.policy.name,
        ),
    )
    evaluated = evaluate_policy(test, chosen.policy, target=target)
    status = "evaluated" if evaluated.bootstrap is not None else "insufficient_evidence"
    return ResearchReport(
        status,
        "Policy locked before heldout evaluation; no automatic activation.",
        target,
        chosen.policy,
        evaluations,
        evaluated,
    )
