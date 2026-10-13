import pytest

from app.tools import calculator as c
from app.tools.calculator import UnitMismatchError


def M(v):
    return {"value": v, "unit": "million"}


def K(v):
    return {"value": v, "unit": "thousand"}


def test_result_shape():
    r = c.percent_change(100, 120)
    assert set(r) == {"name", "formula", "inputs", "result", "unit"}
    assert r["result"] == 20.0 and r["unit"] == "%"


def test_percent_change():
    assert c.percent_change(100, 150)["result"] == 50.0
    assert c.percent_change(200, 100)["result"] == -50.0  # negative change
    assert c.percent_change(-100, -50)["result"] == 50.0  # abs(old)
    assert c.percent_change(5, 5)["result"] == 0.0
    with pytest.raises(ValueError):
        c.percent_change(0, 10)


def test_ratio_and_zero():
    assert c.ratio(1, 3)["result"] == 0.3333  # rounded to 4 decimals
    assert c.ratio(0, 5)["result"] == 0.0
    with pytest.raises(ValueError):
        c.ratio(1, 0)


def test_margin():
    assert c.margin(25, 200)["result"] == 12.5
    assert c.margin(-20, 200)["result"] == -10.0
    with pytest.raises(ValueError):
        c.margin(5, 0)


def test_cagr():
    assert c.cagr(100, 121, 2)["result"] == 10.0
    assert c.cagr(100, 50, 1)["result"] == -50.0
    assert c.cagr(100, 0, 3)["result"] == -100.0
    for bad in [(0, 10, 2), (-5, 10, 2), (100, 150, 0), (100, 150, -1), (100, -1, 2)]:
        with pytest.raises(ValueError):
            c.cagr(*bad)


def test_difference_total_average():
    assert c.difference(10, 25)["result"] == -15.0
    assert c.total([1, 2, 3.5])["result"] == 6.5
    assert c.average([2, 4, 9])["result"] == 5.0
    assert c.total([-5, 5])["result"] == 0.0
    for bad in ([], None, "abc"):
        with pytest.raises(ValueError):
            c.total(bad)
        with pytest.raises(ValueError):
            c.average(bad)


def test_balance_sheet_ratios():
    assert c.debt_to_equity(50, 100)["result"] == 0.5
    assert c.current_ratio(300, 150)["result"] == 2.0
    assert c.eps(1000, 500)["result"] == 2.0
    for f, args in [
        (c.debt_to_equity, (1, 0)),
        (c.current_ratio, (1, 0)),
        (c.eps, (1, 0)),
    ]:
        with pytest.raises(ValueError):
            f(*args)


def test_missing_input():
    with pytest.raises(ValueError):
        c.ratio(None, 2)
    with pytest.raises(ValueError):
        c.ratio(1, "x")
    with pytest.raises(ValueError):
        c.ratio(True, 2)


def test_unit_normalisation_million_vs_thousand():
    assert c.difference(M(5), K(2000))["result"] == 3.0  # 5m - 2m, in millions
    assert c.difference(M(5), K(2000))["unit"] == "million"
    assert c.percent_change(M(1), K(1500))["result"] == 50.0
    assert c.total([M(1), K(500), M(1)])["result"] == 2.5
    assert c.margin(K(500), M(10))["result"] == 5.0
    assert c.eps(M(10), M(5))["result"] == 2.0


def test_unit_mismatch_errors():
    with pytest.raises(UnitMismatchError):
        c.difference(M(5), 3)
    with pytest.raises(UnitMismatchError):
        c.ratio(M(5), {"value": 3, "unit": "%"})
    with pytest.raises(UnitMismatchError):
        c.total(
            [{"value": 1, "unit": "USD million"}, {"value": 1, "unit": "EUR million"}]
        )
    assert c.total([{"value": 1, "unit": "USD million"}, M(1)])["result"] == 2.0


def test_run_tool():
    assert c.run_tool("margin", {"profit": 10, "revenue": 100})["result"] == 10.0
    with pytest.raises(ValueError):
        c.run_tool("margin", {"profit": 10})
    with pytest.raises(ValueError):
        c.run_tool("exec", {})


def test_no_eval_anywhere():
    import inspect

    src = inspect.getsource(c)
    assert "eval(" not in src and "exec(" not in src
