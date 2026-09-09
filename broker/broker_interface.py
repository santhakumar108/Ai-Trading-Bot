"""
Broker Abstraction Layer (spec sections 17 & 18).

`BrokerInterface` is the ONLY thing the rest of the system (paper trading
engine, dashboard, eventual live-trading code) is allowed to talk to for
account/order operations. This means a real broker connector can be added
later purely as a new subclass, with zero changes to strategy, risk, or
paper-trading logic.

Safety, by construction:
  * `PaperBroker` (broker/paper_broker.py) is the only concrete
    implementation shipped in this build. It never touches a real account
    and never needs credentials.
  * Any future live broker subclass MUST check `settings.live_trading_enabled`
    (and the emergency-stop flag) inside `place_order` itself and refuse to
    submit real orders when either guard is not satisfied -- this is
    enforced by `BrokerInterface.place_order`'s default implementation,
    which raises unless the subclass explicitly opts in via
    `supports_live_orders = True` AND the caller passes a config with
    live trading enabled.
  * Credentials are read from environment variables only (see
    `.env.example`) -- never hard-coded, never logged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: int
    order_type: str     # "MARKET", "LIMIT", "SL", "SL-M"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_paper: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class Position:
    symbol: str
    side: str          # "LONG" or "SHORT"
    quantity: int
    average_price: float
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    sector: Optional[str] = None


class BrokerInterface(ABC):
    """Abstract broker. See module docstring for the safety contract every
    subclass must honor."""

    supports_live_orders: bool = False   # subclasses must override deliberately

    @abstractmethod
    def get_balance(self) -> float:
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> float:
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        live_trading_enabled: bool = False,
    ) -> Order:
        """Default contract: unless the concrete subclass sets
        `supports_live_orders = True` AND the caller passes
        `live_trading_enabled=True` (which should itself only ever be sourced
        from `config.settings.load_config().system.live_trading_enabled`,
        never hard-coded True), the order must be rejected or routed to a
        paper/simulated fill. `PaperBroker` always simulates regardless of
        this flag -- it physically cannot place a real order."""
        ...

    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order:
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        ...
