"""
Risk Engine
===========
Two responsibilities, kept separate on purpose:

1. TradeRiskCalculator -- per-trade math. Given entry/stop/target and
   capital, computes risk per share, reward per share, risk/reward ratio,
   position size, maximum possible loss, and expected costs. Position size
   is derived ONLY from capital, max allowed risk, entry, and stop-loss --
   never from confidence. A more confident trade gets a better setup
   requirement, not a bigger, riskier bet.

2. CapitalProtection -- account-level state machine. Tracks daily/weekly
   P&L, open position count, and sector exposure; refuses new trades once
   a loss limit is breached or an emergency stop is set. This state
   persists across calls (it is meant to be held by the paper-trading / live
   engine for the life of a trading day/week).

Nothing here decides whether a setup is *good* (that's technical/news/
signal analysis) -- this module only decides whether a setup that has
already been proposed is *sized and bounded safely*, and whether the
account is currently allowed to take on any new risk at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Per-trade risk/reward + position sizing
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessment:
    symbol: str
    side: str                     # "BUY" or "SELL" (short)
    entry: float
    stop_loss: float
    target: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    position_size: int            # shares/units, floored to whole units -- the
                                   # FINAL executable quantity (risk budget AND
                                   # notional-cap AND cash all satisfied)
    capital_at_risk: float        # position_size * risk_per_share
    max_possible_loss: float      # capital_at_risk (assuming stop holds)
    notional_exposure: float      # position_size * entry
    expected_transaction_cost: float
    expected_slippage_cost: float
    net_expected_reward_per_share: float   # reward_per_share - costs/share
    net_risk_reward_ratio: float           # RR after costs
    approved: bool
    rejection_reasons: List[str] = field(default_factory=list)
    # Expected value (spec section 10: "do not optimize only for win rate").
    # assumed_win_probability defaults to a conservative 0.5 (coin flip)
    # unless a calibrated win-rate-by-confidence function was supplied to
    # TradeRiskCalculator -- see its docstring. This is NOT a guarantee: a
    # positive expected_value_per_share here reflects the ASSUMED
    # probability, not a proven one.
    assumed_win_probability: float = 0.5
    expected_value_per_share: float = 0.0
    expected_value_total: float = 0.0
    expected_value_positive: bool = False
    # Spec Part 10-11: "calculate whether at least ONE whole share can
    # actually be purchased" -- a distinct, PRE-risk-budget question from
    # position_size above. This is the max whole-share quantity affordable
    # by cash alone (entry + estimated round-trip costs), ignoring the
    # risk-per-trade/notional caps. affordable_quantity == 0 means the
    # account cannot afford this instrument at all; affordable_quantity > 0
    # but position_size == 0 means it CAN be bought by cash but the
    # resulting per-share risk exceeds the account's risk budget -- two
    # different reasons for "no trade", reported distinctly (see evaluate()).
    affordable_quantity: int = 0
    transaction_cost_pct_of_capital: float = 0.0  # (txn cost + slippage) / capital -- spec Part 13

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry": round(self.entry, 4),
            "stop_loss": round(self.stop_loss, 4),
            "target": round(self.target, 4),
            "risk_per_share": round(self.risk_per_share, 4),
            "reward_per_share": round(self.reward_per_share, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "net_risk_reward_ratio": round(self.net_risk_reward_ratio, 2),
            "position_size": self.position_size,
            "affordable_quantity": self.affordable_quantity,
            "capital_at_risk": round(self.capital_at_risk, 2),
            "max_possible_loss": round(self.max_possible_loss, 2),
            "notional_exposure": round(self.notional_exposure, 2),
            "expected_transaction_cost": round(self.expected_transaction_cost, 2),
            "expected_slippage_cost": round(self.expected_slippage_cost, 2),
            "transaction_cost_pct_of_capital": round(self.transaction_cost_pct_of_capital, 4),
            "assumed_win_probability": round(self.assumed_win_probability, 3),
            "expected_value_per_share": round(self.expected_value_per_share, 4),
            "expected_value_total": round(self.expected_value_total, 2),
            "expected_value_positive": self.expected_value_positive,
            "approved": self.approved,
            "rejection_reasons": self.rejection_reasons,
        }


class TradeRiskCalculator:
    def __init__(
        self,
        risk_per_trade_pct: float,
        min_risk_reward: float,
        transaction_cost_pct: float,
        slippage_pct: float,
        min_edge_after_costs_pct: float,
        max_single_position_pct: float,
        default_win_probability: float = 0.5,
        min_expected_value_per_share: float = 0.0,
        win_probability_fn: Optional[Callable[[float], float]] = None,
        small_account_capital_threshold: float = 25_000.0,
        small_account_risk_per_trade_pct: float = 0.02,
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_risk_reward = min_risk_reward
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.min_edge_after_costs_pct = min_edge_after_costs_pct
        self.max_single_position_pct = max_single_position_pct
        self.default_win_probability = default_win_probability
        self.min_expected_value_per_share = min_expected_value_per_share
        # Optional calibrated hook: confidence (0-100) -> assumed win
        # probability (0-1), e.g. derived from walk-forward validation's
        # "trades scored 80-89 historically won X% of the time" buckets.
        # Deliberately NOT wired to anything by default -- confidence is a
        # weighted score, not a probability of profit, until a real
        # out-of-sample calibration exercise says otherwise (see
        # strategy/signal_engine.py's module docstring).
        self.win_probability_fn = win_probability_fn
        # Small-account mode (spec Part 10-14) -- see config/settings.py's
        # RiskConfig docstring for the rationale.
        self.small_account_capital_threshold = small_account_capital_threshold
        self.small_account_risk_per_trade_pct = small_account_risk_per_trade_pct

    def _effective_risk_pct(self, capital: float, risk_multiplier: float) -> float:
        """
        Resolves the risk-per-trade percentage to use for THIS evaluation,
        decided purely from account size and the caller-supplied
        risk_multiplier (spec Part 26 loss-streak reduction) -- NEVER from
        whether a specific candidate would otherwise pass or fail. Both
        inputs are fixed before any candidate-specific math (entry/stop/
        target) runs, so this can't be read as "raise risk until the trade
        goes through" (spec Part 11 explicitly forbids that).
        """
        base_pct = (
            self.small_account_risk_per_trade_pct
            if capital < self.small_account_capital_threshold
            else self.risk_per_trade_pct
        )
        return base_pct * risk_multiplier

    def evaluate(
        self,
        symbol: str,
        side: str,
        entry: float,
        stop_loss: float,
        target: float,
        capital: float,
        confidence: Optional[float] = None,
        risk_multiplier: float = 1.0,
    ) -> RiskAssessment:
        reasons: List[str] = []

        if entry <= 0 or stop_loss <= 0 or target <= 0:
            reasons.append("Entry/stop/target must be positive prices.")
            return self._rejected(symbol, side, entry, stop_loss, target, reasons)

        if side.upper() == "BUY":
            risk_per_share = entry - stop_loss
            reward_per_share = target - entry
        elif side.upper() == "SELL":
            risk_per_share = stop_loss - entry
            reward_per_share = entry - target
        else:
            reasons.append(f"Unknown side '{side}'.")
            return self._rejected(symbol, side, entry, stop_loss, target, reasons)

        if risk_per_share <= 0:
            reasons.append(
                "Stop-loss is not logically defined (it does not sit on the "
                "risk side of entry) -- refusing to size this trade."
            )
            return self._rejected(symbol, side, entry, stop_loss, target, reasons)

        if reward_per_share <= 0:
            reasons.append("Target does not offer positive reward relative to entry.")
            return self._rejected(symbol, side, entry, stop_loss, target, reasons)

        risk_reward_ratio = reward_per_share / risk_per_share

        # Transaction costs & slippage: approximate as a percentage of
        # notional, applied on both entry and exit (two legs).
        cost_per_share = entry * self.transaction_cost_pct
        slippage_per_share = entry * self.slippage_pct
        total_cost_per_share = (cost_per_share + slippage_per_share) * 2  # entry + exit

        net_reward_per_share = reward_per_share - total_cost_per_share
        net_risk_reward_ratio = net_reward_per_share / risk_per_share if risk_per_share else 0.0

        if net_reward_per_share <= 0:
            reasons.append("Transaction costs and slippage consume the entire expected reward.")
        edge_pct_of_price = net_reward_per_share / entry if entry else 0
        if edge_pct_of_price < self.min_edge_after_costs_pct:
            reasons.append(
                f"Expected edge after costs ({edge_pct_of_price:.3%}) is below the "
                f"configured minimum ({self.min_edge_after_costs_pct:.3%})."
            )

        if risk_reward_ratio < self.min_risk_reward:
            reasons.append(
                f"Risk/reward {risk_reward_ratio:.2f} is below the minimum required "
                f"{self.min_risk_reward:.2f}."
            )

        # Expected value after costs (spec section 10: "do not optimize only
        # for win rate"). Costs are charged on BOTH outcomes here (a losing
        # trade still pays brokerage/slippage on entry and exit), unlike
        # net_risk_reward_ratio above which only nets costs off the reward
        # leg -- this is deliberately the more conservative of the two.
        # win_probability is the ASSUMED probability (see class docstring);
        # a positive expected_value_per_share is a conditional statement
        # ("if this trade really does win at that rate"), never a promise.
        if self.win_probability_fn is not None and confidence is not None:
            win_probability = float(np.clip(self.win_probability_fn(confidence), 0.0, 1.0))
        else:
            win_probability = float(np.clip(self.default_win_probability, 0.0, 1.0))
        net_loss_per_share = risk_per_share + total_cost_per_share
        expected_value_per_share = (
            win_probability * net_reward_per_share - (1 - win_probability) * net_loss_per_share
        )
        expected_value_positive = expected_value_per_share >= self.min_expected_value_per_share
        if not expected_value_positive:
            reasons.append(
                f"Expected value after costs ({expected_value_per_share:.4f}/share at an assumed "
                f"{win_probability:.0%} win probability) is below the minimum required "
                f"({self.min_expected_value_per_share:.4f}) -- this is a conservative estimate, "
                f"not proof the setup loses money; do not treat it as more certain than that."
            )

        # Position sizing: capital-and-stop-distance driven, NOT confidence driven.
        effective_risk_pct = self._effective_risk_pct(capital, risk_multiplier)
        max_risk_capital = capital * effective_risk_pct
        position_size = int(max_risk_capital // risk_per_share) if risk_per_share > 0 else 0

        # Cap by max single-position notional exposure regardless of stop distance.
        max_notional = capital * self.max_single_position_pct
        if entry > 0:
            max_shares_by_notional = int(max_notional // entry)
            position_size = min(position_size, max_shares_by_notional)

        # Spec Part 10-11: whether at least one whole share can actually be
        # purchased is a distinct question from the risk-budget sizing above
        # -- computed by CASH alone (entry + round-trip transaction costs/
        # slippage), never by risk-per-trade %. This never expands
        # position_size; it only explains WHY position_size came out zero.
        cost_and_entry_per_share = entry + total_cost_per_share
        affordable_quantity = int(capital // cost_and_entry_per_share) if cost_and_entry_per_share > 0 else 0
        # The final executable quantity can never exceed what cash alone
        # affords, regardless of what the risk-budget/notional math allowed.
        position_size = min(position_size, affordable_quantity)

        if position_size <= 0:
            if affordable_quantity <= 0:
                reasons.append(
                    f"INSUFFICIENT CAPITAL: cannot purchase even one share of {symbol} at "
                    f"{entry:.2f} plus estimated round-trip costs ({total_cost_per_share:.2f}/share) "
                    f"with available capital of {capital:.2f}."
                )
            else:
                reasons.append(
                    f"One share's risk ({risk_per_share:.2f}) exceeds the account's per-trade risk "
                    f"budget ({max_risk_capital:.2f} = {effective_risk_pct:.2%} of {capital:.2f}); "
                    f"{affordable_quantity} share(s) are affordable by cash but refusing to trade -- "
                    f"never secretly increasing the risk limit to force a share through."
                )

        capital_at_risk = position_size * risk_per_share
        notional_exposure = position_size * entry
        expected_transaction_cost = cost_per_share * 2 * position_size
        expected_slippage_cost = slippage_per_share * 2 * position_size
        transaction_cost_pct_of_capital = (
            (expected_transaction_cost + expected_slippage_cost) / capital if capital > 0 else 0.0
        )
        approved = len(reasons) == 0 and position_size > 0

        return RiskAssessment(
            symbol=symbol,
            side=side.upper(),
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=risk_reward_ratio,
            position_size=position_size,
            affordable_quantity=affordable_quantity,
            capital_at_risk=capital_at_risk,
            max_possible_loss=capital_at_risk,
            notional_exposure=notional_exposure,
            expected_transaction_cost=expected_transaction_cost,
            expected_slippage_cost=expected_slippage_cost,
            transaction_cost_pct_of_capital=transaction_cost_pct_of_capital,
            net_expected_reward_per_share=net_reward_per_share,
            net_risk_reward_ratio=net_risk_reward_ratio,
            approved=approved,
            rejection_reasons=reasons,
            assumed_win_probability=win_probability,
            expected_value_per_share=expected_value_per_share,
            expected_value_total=expected_value_per_share * position_size,
            expected_value_positive=expected_value_positive,
        )

    def _rejected(self, symbol, side, entry, stop_loss, target, reasons) -> RiskAssessment:
        return RiskAssessment(
            symbol=symbol, side=side, entry=entry, stop_loss=stop_loss, target=target,
            risk_per_share=0.0, reward_per_share=0.0, risk_reward_ratio=0.0,
            position_size=0, capital_at_risk=0.0, max_possible_loss=0.0,
            notional_exposure=0.0, expected_transaction_cost=0.0,
            expected_slippage_cost=0.0, net_expected_reward_per_share=0.0,
            net_risk_reward_ratio=0.0, approved=False, rejection_reasons=reasons,
            assumed_win_probability=self.default_win_probability,
            expected_value_per_share=0.0, expected_value_total=0.0, expected_value_positive=False,
        )


# ---------------------------------------------------------------------------
# 2. Account-level capital protection
# ---------------------------------------------------------------------------


@dataclass
class CapitalState:
    starting_capital: float
    current_capital: float
    day: date
    week_start: date
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    open_positions: int = 0
    sector_exposure: Dict[str, float] = field(default_factory=dict)  # sector -> notional
    trading_halted: bool = False
    halt_reason: str = ""
    # Spec Part 26: consecutive LOSING closed trades, reset to 0 on any win.
    # Read by CapitalProtection.current_risk_multiplier() to shrink (never
    # grow) the effective risk-per-trade after a losing streak.
    consecutive_losses: int = 0


class CapitalProtection:
    """
    Stateful account guardrail. Call `register_fill` / `register_close` as
    trades happen, and `pre_trade_check` before sizing any new trade.
    Once a daily/weekly loss limit is breached, ALL new trades are refused
    until the next day/week (or until the state is manually reset), even if
    the very next setup looks perfect. This is deliberate: the point of a
    loss limit is that it does not bend for a "good enough" trade.
    """

    def __init__(
        self,
        starting_capital: float,
        max_daily_loss_pct: float,
        max_weekly_loss_pct: float,
        max_simultaneous_positions: int,
        max_sector_exposure_pct: float,
        max_single_position_pct: float,
        today: Optional[date] = None,
        max_consecutive_losses_before_reduction: int = 3,
        risk_reduction_factor: float = 0.5,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_simultaneous_positions = max_simultaneous_positions
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_single_position_pct = max_single_position_pct
        self.max_consecutive_losses_before_reduction = max_consecutive_losses_before_reduction
        self.risk_reduction_factor = risk_reduction_factor

        today = today or date.today()
        self.state = CapitalState(
            starting_capital=starting_capital,
            current_capital=starting_capital,
            day=today,
            week_start=today - timedelta(days=today.weekday()),
        )

    def _roll_periods(self, today: date) -> None:
        if today != self.state.day:
            self.state.daily_pnl = 0.0
            self.state.day = today
        week_start = today - timedelta(days=today.weekday())
        if week_start != self.state.week_start:
            self.state.weekly_pnl = 0.0
            self.state.week_start = week_start

    def pre_trade_check(
        self,
        symbol: str,
        sector: Optional[str],
        notional_exposure: float,
        today: Optional[date] = None,
    ) -> List[str]:
        """Returns a list of reasons a new trade must be refused. Empty list
        means the account-level checks pass (individual trade risk/reward
        checks are separate, in TradeRiskCalculator)."""
        today = today or date.today()
        self._roll_periods(today)
        reasons: List[str] = []

        if self.state.trading_halted:
            reasons.append(f"Trading halted: {self.state.halt_reason}")

        daily_loss_limit = -self.state.starting_capital * self.max_daily_loss_pct
        if self.state.daily_pnl <= daily_loss_limit:
            reasons.append(
                f"Daily loss limit reached ({self.state.daily_pnl:.2f} <= "
                f"{daily_loss_limit:.2f}). No new trades until tomorrow."
            )

        weekly_loss_limit = -self.state.starting_capital * self.max_weekly_loss_pct
        if self.state.weekly_pnl <= weekly_loss_limit:
            reasons.append(
                f"Weekly loss limit reached ({self.state.weekly_pnl:.2f} <= "
                f"{weekly_loss_limit:.2f}). No new trades until next week."
            )

        if self.state.open_positions >= self.max_simultaneous_positions:
            reasons.append(
                f"Max simultaneous positions reached ({self.state.open_positions}/"
                f"{self.max_simultaneous_positions})."
            )

        if sector:
            current_sector_notional = self.state.sector_exposure.get(sector, 0.0)
            projected = current_sector_notional + notional_exposure
            max_sector_notional = self.state.current_capital * self.max_sector_exposure_pct
            if projected > max_sector_notional:
                reasons.append(
                    f"Sector exposure limit exceeded for '{sector}' "
                    f"({projected:.2f} > {max_sector_notional:.2f})."
                )

        max_position_notional = self.state.current_capital * self.max_single_position_pct
        if notional_exposure > max_position_notional:
            reasons.append(
                f"Single-position exposure ({notional_exposure:.2f}) exceeds max allowed "
                f"({max_position_notional:.2f})."
            )

        return reasons

    def register_open(self, sector: Optional[str], notional_exposure: float) -> None:
        self.state.open_positions += 1
        if sector:
            self.state.sector_exposure[sector] = self.state.sector_exposure.get(sector, 0.0) + notional_exposure

    def register_close(self, sector: Optional[str], notional_exposure: float, realized_pnl: float, today: Optional[date] = None) -> None:
        today = today or date.today()
        self._roll_periods(today)
        self.state.open_positions = max(0, self.state.open_positions - 1)
        if sector:
            self.state.sector_exposure[sector] = max(0.0, self.state.sector_exposure.get(sector, 0.0) - notional_exposure)
        self.state.daily_pnl += realized_pnl
        self.state.weekly_pnl += realized_pnl
        self.state.current_capital += realized_pnl

        # Spec Part 26: track consecutive losses (reset on any winning close).
        # A realized_pnl of exactly 0.0 is treated as a non-loss (does not
        # extend the streak) but also does not reset it -- a breakeven trade
        # is neither evidence the streak continued nor that it's over.
        if realized_pnl < 0:
            self.state.consecutive_losses += 1
        elif realized_pnl > 0:
            self.state.consecutive_losses = 0

        daily_loss_limit = -self.state.starting_capital * self.max_daily_loss_pct
        if self.state.daily_pnl <= daily_loss_limit:
            self.halt("Daily loss limit breached.")

    def current_risk_multiplier(self) -> float:
        """
        Spec Part 26: after `max_consecutive_losses_before_reduction`
        consecutive losing trades, shrink the effective risk-per-trade by
        `risk_reduction_factor` for subsequent evaluations (see
        risk/risk_engine.py's TradeRiskCalculator._effective_risk_pct()).
        Returns 1.0 (no reduction) otherwise. This can only ever REDUCE the
        risk used by the NEXT trade -- it never increases it and never
        mutates any configured threshold.
        """
        if self.state.consecutive_losses >= self.max_consecutive_losses_before_reduction:
            return self.risk_reduction_factor
        return 1.0

    def halt(self, reason: str) -> None:
        self.state.trading_halted = True
        self.state.halt_reason = reason

    def emergency_stop(self, reason: str = "Manual emergency stop triggered.") -> None:
        self.halt(reason)

    def resume(self) -> None:
        self.state.trading_halted = False
        self.state.halt_reason = ""

    def detect_abnormal_market(self, index_daily_return: float, threshold: float = 0.05) -> bool:
        """A simple, transparent circuit breaker: if the benchmark index
        moved more than `threshold` in a single session, treat conditions
        as abnormal and halt new trades until a human resumes trading."""
        if abs(index_daily_return) >= threshold:
            self.halt(f"Abnormal market move detected (index moved {index_daily_return:.2%}).")
            return True
        return False
