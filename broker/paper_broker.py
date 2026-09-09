"""
Paper (simulated) broker.

The only broker implementation shipped in this build. Never sends a real
order anywhere -- `place_order` always simulates a fill against the current
quote (from a MarketDataProvider), applies configurable slippage and fees,
and updates an in-memory virtual ledger. This is what section 18's
"LIVE_TRADING_ENABLED must default to False and be enforced" looks like in
practice: there is currently no code path in this class that can reach a
real exchange, regardless of how that flag is set.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from broker.broker_interface import BrokerInterface, Order, OrderStatus, Position
from data.market_data import MarketDataProvider


class PaperBroker(BrokerInterface):
    supports_live_orders = False  # physically incapable of live orders

    def __init__(
        self,
        starting_capital: float,
        market_data: Optional[MarketDataProvider] = None,
        slippage_pct: float = 0.0007,
        fee_pct: float = 0.0010,
    ):
        self._cash = starting_capital
        self.starting_capital = starting_capital
        self.market_data = market_data or MarketDataProvider()
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}

    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        return self._cash

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_quote(self, symbol: str) -> float:
        return self.market_data.get_quote(symbol)

    # ------------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        live_trading_enabled: bool = False,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        sector: Optional[str] = None,
    ) -> Order:
        # live_trading_enabled is accepted for interface compatibility but is
        # deliberately IGNORED here: PaperBroker never places a real order,
        # no matter what the caller passes.
        order_id = str(uuid.uuid4())

        if quantity <= 0:
            order = Order(order_id=order_id, symbol=symbol, side=side, quantity=quantity,
                           order_type=order_type, status=OrderStatus.REJECTED,
                           rejection_reason="Quantity must be positive.")
            self._orders[order_id] = order
            return order

        try:
            quote = self.get_quote(symbol)
        except Exception as exc:
            order = Order(order_id=order_id, symbol=symbol, side=side, quantity=quantity,
                           order_type=order_type, status=OrderStatus.REJECTED,
                           rejection_reason=f"Market data unavailable: {exc}")
            self._orders[order_id] = order
            return order

        slip = quote * self.slippage_pct
        fill_price = quote + slip if side.upper() == "BUY" else quote - slip

        # A BUY against an existing SHORT (or a SELL against an existing
        # LONG) is a CLOSE/COVER, not a fresh open -- must reduce/remove
        # the existing opposite-side position, never create an unrelated
        # new one on top of it. Fee/notional are charged on the quantity
        # actually closed (never more than what's held), and the returned
        # Order reports that real filled quantity, not the requested one.
        existing = self._positions.get(symbol)
        is_closing = existing is not None and (
            (side.upper() == "BUY" and existing.side == "SHORT")
            or (side.upper() == "SELL" and existing.side == "LONG")
        )

        if is_closing:
            close_qty = min(quantity, existing.quantity)
            notional = fill_price * close_qty
            fee = notional * self.fee_pct

            if side.upper() == "BUY":  # covering a SHORT
                total_cost = notional + fee
                if total_cost > self._cash:
                    order = Order(order_id=order_id, symbol=symbol, side=side, quantity=quantity,
                                   order_type=order_type, status=OrderStatus.REJECTED,
                                   rejection_reason="Insufficient virtual capital.")
                    self._orders[order_id] = order
                    return order
                self._cash -= total_cost
            else:  # SELL closing a LONG
                self._cash += notional - fee

            existing.quantity -= close_qty
            if existing.quantity <= 0:
                del self._positions[symbol]
            filled_quantity = close_qty
        else:
            notional = fill_price * quantity
            fee = notional * self.fee_pct

            if side.upper() == "BUY":  # fresh open, or adding to an existing LONG
                total_cost = notional + fee
                if total_cost > self._cash:
                    order = Order(order_id=order_id, symbol=symbol, side=side, quantity=quantity,
                                   order_type=order_type, status=OrderStatus.REJECTED,
                                   rejection_reason="Insufficient virtual capital.")
                    self._orders[order_id] = order
                    return order
                self._cash -= total_cost
                if existing and existing.side == "LONG":
                    total_qty = existing.quantity + quantity
                    existing.average_price = (existing.average_price * existing.quantity + fill_price * quantity) / total_qty
                    existing.quantity = total_qty
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol, side="LONG", quantity=quantity, average_price=fill_price,
                        stop_loss=stop_loss, target=target, sector=sector,
                    )
            else:  # SELL -- fresh open, or adding to an existing SHORT (paper only)
                self._cash += notional - fee
                if existing and existing.side == "SHORT":
                    total_qty = existing.quantity + quantity
                    existing.average_price = (existing.average_price * existing.quantity + fill_price * quantity) / total_qty
                    existing.quantity = total_qty
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol, side="SHORT", quantity=quantity, average_price=fill_price,
                        stop_loss=stop_loss, target=target, sector=sector,
                    )
            filled_quantity = quantity

        order = Order(
            order_id=order_id, symbol=symbol, side=side, quantity=quantity, order_type=order_type,
            limit_price=limit_price, stop_price=stop_price, status=OrderStatus.FILLED,
            filled_price=fill_price, filled_quantity=filled_quantity, is_paper=True,
        )
        self._orders[order_id] = order
        return order

    def modify_order(self, order_id: str, **kwargs) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order_id {order_id}")
        if order.status == OrderStatus.FILLED:
            raise ValueError("Cannot modify an already-filled paper order.")
        for k, v in kwargs.items():
            if hasattr(order, k):
                setattr(order, k, v)
        order.updated_at = datetime.utcnow()
        return order

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order_id {order_id}")
        if order.status == OrderStatus.FILLED:
            raise ValueError("Cannot cancel an already-filled paper order.")
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return order

    def get_order_status(self, order_id: str) -> OrderStatus:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order_id {order_id}")
        return order.status

    def equity(self) -> float:
        """Cash + mark-to-market value of open positions. For a SHORT,
        `self._cash` already holds the sale proceeds (credited when the
        short was opened) -- the only remaining contribution is the
        negative liability to buy it back at the current quote."""
        total = self._cash
        for pos in self._positions.values():
            try:
                quote = self.get_quote(pos.symbol)
            except Exception:
                quote = pos.average_price
            if pos.side == "LONG":
                total += quote * pos.quantity
            else:
                total -= quote * pos.quantity
        return total

    def unrealized_pnl(self) -> float:
        """Sum of (mark-to-market - entry) P&L across open positions only
        -- excludes cash, unlike equity(). LONG: (quote-avg)*qty; SHORT:
        (avg-quote)*qty."""
        total = 0.0
        for pos in self._positions.values():
            try:
                quote = self.get_quote(pos.symbol)
            except Exception:
                quote = pos.average_price
            if pos.side == "LONG":
                total += (quote - pos.average_price) * pos.quantity
            else:
                total += (pos.average_price - quote) * pos.quantity
        return total
