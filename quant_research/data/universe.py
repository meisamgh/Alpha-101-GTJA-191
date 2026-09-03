"""Universe definitions with explicit survivorship metadata."""
from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    point_in_time: bool
    liquidity_lookback: int = 60
    max_assets: int = 500

    def disclosure(self) -> str:
        return ("point-in-time membership" if self.point_in_time else
                "current/static membership; results are subject to survivorship bias")
