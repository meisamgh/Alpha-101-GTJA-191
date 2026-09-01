"""Trusted-factor registry: only formula-tested, trailing factors are enabled."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustedFactor:
    name: str
    family: str
    formula_reference: str
    trailing_only: bool = True


TRUSTED_FACTORS = {
    factor.name: factor
    for factor in (
        TrustedFactor("alpha_012", "Alpha101", "sign(delta(volume,1)) * -delta(close,1)"),
        TrustedFactor("alpha_101", "Alpha101", "(close-open)/(high-low+0.001)"),
        TrustedFactor("gtja_002", "GTJA191", "-delta(((C-L)-(H-C))/(H-L),1)"),
        TrustedFactor("gtja_012", "GTJA191", "-rank(open-mean(vwap,10))*rank(abs(C-vwap))"),
    )
}
