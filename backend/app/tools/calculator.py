"""Safe finance calculator. Plain Python maths only (no eval); the LLM never does arithmetic.

Every input may be a plain number or a dict {"value": 12.5, "unit": "million"}.
Values with scale units (thousand / million / billion) are normalised before calculating,
so a value in millions is never mixed with one in thousands. If the units cannot be
normalised (one side has a scale and the other has none, or the currencies differ)
a UnitMismatchError (a ValueError) is raised.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SCALES: dict[str, float] = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
}
CURRENCY_ALIASES: dict[str, str] = {
    "$": "usd",
    "us$": "usd",
    "dollar": "usd",
    "dollars": "usd",
}
DECIMALS = 4


class UnitMismatchError(ValueError):
    """Raised when inputs have units that cannot be put on the same scale."""


@dataclass
class _Quantity:
    value: float
    scale: float | None  # None = no unit given
    currency: str  # e.g. "usd", "%", or "" when unspecified
    unit: str | None  # the original unit text

    @property
    def base(self) -> float:
        """Value in base units (e.g. 5 million -> 5_000_000)."""
        return self.value * (self.scale or 1.0)


def _quantity(x: Any, name: str) -> _Quantity:
    unit: str | None = None
    if isinstance(x, dict):
        unit = x.get("unit")
        x = x.get("value")
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"Missing or non-numeric input: {name}")
    if not unit or not str(unit).strip():
        return _Quantity(float(x), None, "", None)
    tokens = [CURRENCY_ALIASES.get(t, t) for t in str(unit).lower().split()]
    if tokens[-1] in SCALES:
        return _Quantity(float(x), SCALES[tokens[-1]], " ".join(tokens[:-1]), str(unit))
    return _Quantity(float(x), 1.0, " ".join(tokens), str(unit))


def _align(quantities: list[_Quantity]) -> None:
    """Raise UnitMismatchError if the quantities cannot share one scale."""
    has_unit = {q.scale is not None for q in quantities}
    if len(has_unit) > 1:
        raise UnitMismatchError(
            "Units differ: some inputs have a unit and some do not."
        )
    if any(q.currency == "%" for q in quantities) and any(
        q.currency != "%" for q in quantities
    ):
        raise UnitMismatchError(
            "Units differ: a percentage cannot be mixed with other units."
        )
    currencies = {q.currency for q in quantities if q.currency}
    if len(currencies) > 1:
        raise UnitMismatchError(f"Units differ: {sorted(currencies)}.")


def _need(**kwargs: Any) -> dict[str, _Quantity]:
    quantities = {k: _quantity(v, k) for k, v in kwargs.items()}
    _align(list(quantities.values()))
    return quantities


def _result(
    name: str, formula: str, inputs: dict[str, Any], result: float, unit: str | None
) -> dict[str, Any]:
    return {
        "name": name,
        "formula": formula,
        "inputs": inputs,
        "result": round(result, DECIMALS),
        "unit": unit,
    }


def _divide(a: float, b: float, what: str) -> float:
    if b == 0:
        raise ValueError(f"Cannot divide by zero ({what}).")
    return a / b


def percent_change(old: Any, new: Any) -> dict[str, Any]:
    q = _need(old=old, new=new)
    base_old, base_new = q["old"].base, q["new"].base
    value = _divide(base_new - base_old, abs(base_old), "old value is zero") * 100
    return _result(
        "percent_change",
        "(new - old) / abs(old) * 100",
        {"old": old, "new": new},
        value,
        "%",
    )


def ratio(a: Any, b: Any) -> dict[str, Any]:
    q = _need(a=a, b=b)
    value = _divide(q["a"].base, q["b"].base, "b is zero")
    return _result("ratio", "a / b", {"a": a, "b": b}, value, "x")


def margin(profit: Any, revenue: Any) -> dict[str, Any]:
    q = _need(profit=profit, revenue=revenue)
    value = _divide(q["profit"].base, q["revenue"].base, "revenue is zero") * 100
    return _result(
        "margin",
        "profit / revenue * 100",
        {"profit": profit, "revenue": revenue},
        value,
        "%",
    )


def cagr(start: Any, end: Any, years: Any) -> dict[str, Any]:
    q = _need(start=start, end=end)
    n = _quantity(years, "years").value
    if n <= 0:
        raise ValueError("years must be greater than zero.")
    s, e = q["start"].base, q["end"].base
    if s <= 0:
        raise ValueError("start must be greater than zero for CAGR.")
    if e < 0:
        raise ValueError("end must not be negative for CAGR.")
    value = ((e / s) ** (1 / n) - 1) * 100
    return _result(
        "cagr",
        "(end / start) ** (1 / years) - 1, as percent",
        {"start": start, "end": end, "years": years},
        value,
        "%",
    )


def difference(a: Any, b: Any) -> dict[str, Any]:
    q = _need(a=a, b=b)
    first = q["a"]
    value = (first.base - q["b"].base) / (first.scale or 1.0)
    return _result("difference", "a - b", {"a": a, "b": b}, value, first.unit)


def _list_quantities(values: Any) -> list[_Quantity]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("values must be a non-empty list.")
    quantities = [_quantity(v, f"values[{i}]") for i, v in enumerate(values)]
    _align(quantities)
    return quantities


def total(values: Any) -> dict[str, Any]:
    qs = _list_quantities(values)
    value = sum(q.base for q in qs) / (qs[0].scale or 1.0)
    return _result(
        "total", "sum of the list", {"values": list(values)}, value, qs[0].unit
    )


def average(values: Any) -> dict[str, Any]:
    qs = _list_quantities(values)
    value = sum(q.base for q in qs) / len(qs) / (qs[0].scale or 1.0)
    return _result(
        "average", "sum / count", {"values": list(values)}, value, qs[0].unit
    )


def debt_to_equity(debt: Any, equity: Any) -> dict[str, Any]:
    q = _need(debt=debt, equity=equity)
    value = _divide(q["debt"].base, q["equity"].base, "equity is zero")
    return _result(
        "debt_to_equity", "debt / equity", {"debt": debt, "equity": equity}, value, "x"
    )


def current_ratio(assets: Any, liabilities: Any) -> dict[str, Any]:
    q = _need(assets=assets, liabilities=liabilities)
    value = _divide(q["assets"].base, q["liabilities"].base, "liabilities are zero")
    return _result(
        "current_ratio",
        "current assets / current liabilities",
        {"assets": assets, "liabilities": liabilities},
        value,
        "x",
    )


def eps(net_income: Any, shares: Any) -> dict[str, Any]:
    q = _need(net_income=net_income, shares=shares)
    value = _divide(q["net_income"].base, q["shares"].base, "shares are zero")
    return _result(
        "eps",
        "net income / shares",
        {"net_income": net_income, "shares": shares},
        value,
        "per share",
    )


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    f.__name__: f
    for f in (
        percent_change,
        ratio,
        margin,
        cagr,
        difference,
        total,
        average,
        debt_to_equity,
        current_ratio,
        eps,
    )
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one calculator function by name. Raises ValueError for unknown names or bad inputs."""
    func = TOOLS.get(name)
    if func is None:
        raise ValueError(f"Unknown calculator function: {name}")
    try:
        return func(**args)
    except TypeError as exc:
        raise ValueError(f"Missing or unexpected input for {name}: {exc}") from None
