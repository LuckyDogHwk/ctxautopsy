# -*- coding: utf-8 -*-
"""模块 B（billing）的 pytest 测试 = 最终接口契约。"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from ctxautopsy.billing import (
    DEFAULT_PRICES,
    compute_cost_range,
    cost_sessions,
    cost_usage,
    costs_to_json,
    is_peak_beijing,
    load_price_table,
)
from ctxautopsy.session_log import SessionReport, UsageTiers, parse_session

DATA = Path(__file__).parent / "data"


def _usage(**kw):
    base = dict(new_input_tokens=0, cache_write_tokens=0, cache_write_5m_tokens=0,
                cache_write_1h_tokens=0, cache_read_tokens=0, output_tokens=0, api_calls=1)
    base.update(kw)
    return UsageTiers(**base)


def test_四档成本计算与总计():
    """claude-sonnet-5-5: 2.0/2.5(5m)/4.0(1h)/0.2/10.0（USD/百万 token）"""
    u = _usage(new_input_tokens=1000, cache_write_tokens=1000, cache_write_5m_tokens=1000,
               cache_read_tokens=100000, output_tokens=2000)
    c = cost_usage(u, "claude-sonnet-5-5", DEFAULT_PRICES)
    assert c.priced
    assert c.currency == "USD"
    tiers = {t.tier: t.cost for t in c.tiers}
    assert tiers["new_input"] == pytest.approx(1000 / 1e6 * 2.0)
    assert tiers["cache_write"] == pytest.approx(1000 / 1e6 * 2.5)
    assert tiers["cache_read"] == pytest.approx(100000 / 1e6 * 0.2)
    assert tiers["output"] == pytest.approx(2000 / 1e6 * 10.0)
    assert c.total_cost == pytest.approx(0.0445)


def test_缓存写分5m1h档计价():
    u = _usage(new_input_tokens=1000, cache_write_tokens=1000,
               cache_write_5m_tokens=600, cache_write_1h_tokens=400,
               cache_read_tokens=100000, output_tokens=2000)
    c = cost_usage(u, "claude-sonnet-5-5", DEFAULT_PRICES)
    tiers = {t.tier: t.cost for t in c.tiers}
    # 5m 档 1.25×、1h 档 2×
    assert tiers["cache_write"] == pytest.approx(600 / 1e6 * 2.5 + 400 / 1e6 * 4.0)
    assert c.total_cost == pytest.approx(0.0451)


def test_未分档缓存写按5m计价并告警():
    u = _usage(cache_write_tokens=1500)  # 无 5m/1h 分档
    c = cost_usage(u, "claude-sonnet-5-5", DEFAULT_PRICES)
    tiers = {t.tier: t.cost for t in c.tiers}
    assert tiers["cache_write"] == pytest.approx(1500 / 1e6 * 2.5)
    assert any("5m" in w for w in c.warnings)


def test_反事实无缓存成本与缓存节省():
    u = _usage(new_input_tokens=1000, cache_write_tokens=1000, cache_write_5m_tokens=1000,
               cache_read_tokens=100000, output_tokens=2000)
    c = cost_usage(u, "claude-sonnet-5-5", DEFAULT_PRICES)
    # 全部输入按新输入价：102000/1e6*2 + 2000/1e6*10
    assert c.cost_if_uncached == pytest.approx(0.224)
    assert c.cache_savings == pytest.approx(0.224 - 0.0445)


def test_未知模型不编价格():
    c = cost_usage(_usage(new_input_tokens=100), "unknown-model-xyz", DEFAULT_PRICES)
    assert not c.priced
    assert c.total_cost is None
    assert c.cost_if_uncached is None
    assert c.cache_savings is None
    assert c.warnings


def test_价格表json覆盖内置行且保留其余行():
    custom = DATA / "custom_prices.json"
    custom.write_text(json.dumps({
        "rows": {
            "claude-sonnet-5-5": {"input": 9.99, "cache_write_5m": 12.0, "cache_write_1h": 19.0,
                                  "cache_read": 0.99, "output": 30.0, "currency": "USD",
                                  "source": "测试覆盖", "note": ""}
        },
    }), encoding="utf-8")
    try:
        table = load_price_table(custom)
        assert table.rows["claude-sonnet-5-5"].input == 9.99
        assert table.rows["claude-opus-5-5"].input == 4.00  # 未覆盖的行保留
    finally:
        custom.unlink()


def test_价格表可从环境变量加载(monkeypatch):
    custom = DATA / "custom_prices.json"
    custom.write_text(json.dumps({
        "rows": {"my-model": {"input": 1.0, "cache_write_5m": 1.0, "cache_write_1h": 1.0,
                              "cache_read": 0.1, "output": 2.0, "currency": "USD"}},
    }), encoding="utf-8")
    monkeypatch.setenv("CTXAUTOPSY_PRICES", str(custom))
    try:
        table = load_price_table(None)  # path=None → 读 CTXAUTOPSY_PRICES 环境变量
        assert table.rows["my-model"].input == 1.0
    finally:
        custom.unlink()


def test_内置价格表每行都有来源标注():
    for model, row in DEFAULT_PRICES.rows.items():
        assert row.source, f"{model} 缺价格来源标注"
        for v in (row.input, row.cache_write_5m, row.cache_write_1h, row.cache_read, row.output):
            assert v is not None and v >= 0, f"{model} 价格字段非法: {v}"
        assert row.currency in ("USD", "CNY")
    assert "claude-sonnet-5-5" in DEFAULT_PRICES.rows
    assert "claude-opus-5-5" in DEFAULT_PRICES.rows
    assert "deepseek-v4-flash" in DEFAULT_PRICES.rows
    assert "deepseek-v4-pro" in DEFAULT_PRICES.rows
    assert "deepseek-v4-pro-peak" in DEFAULT_PRICES.rows  # 高峰参考行


def test_坏价格表文件报中文错误():
    bad = DATA / "bad_prices.json"
    bad.write_text("{not json", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="价格表"):
            load_price_table(bad)
    finally:
        bad.unlink()


def test_价格表缺字段报中文错误():
    bad = DATA / "bad_prices.json"
    bad.write_text(json.dumps({"rows": {"m": {"input": 1.0}}}), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="缺字段"):
            load_price_table(bad)
    finally:
        bad.unlink()


def test_cost_sessions_汇总与harness自报对比():
    r = parse_session(DATA / "sample_new_only.jsonl")
    project = cost_sessions([r], DEFAULT_PRICES)
    assert project.totals_by_currency["USD"] > 0
    assert project.total_cost == pytest.approx(project.totals_by_currency["USD"])
    # 推算 0.063 vs 自报 0.123 USD
    assert project.delta_vs_self_reported == pytest.approx((0.063 - 0.123) / 0.123)
    assert "claude-sonnet-5-5" in project.by_model


def test_cost_sessions_未定价模型进unpriced清单():
    r = parse_session(DATA / "sample_total_semantics.jsonl")  # model "m-x"
    project = cost_sessions([r], DEFAULT_PRICES)
    assert "m-x" in project.unpriced_models
    assert project.delta_vs_self_reported is None


def test_cost_sessions_币种混杂不假求和():
    custom = DATA / "custom_prices.json"
    custom.write_text(json.dumps({
        "rows": {"deepseek-v4-flash": {"input": 1.0, "cache_write_5m": 1.0, "cache_write_1h": 1.0,
                                       "cache_read": 0.02, "output": 4.0, "currency": "CNY"}},
    }), encoding="utf-8")
    try:
        table = load_price_table(custom)
        usd = SessionReport(session_id="u",
                            per_model={"claude-sonnet-5-5": _usage(new_input_tokens=1_000_000)})
        cny = SessionReport(session_id="c",
                            per_model={"deepseek-v4-flash": _usage(new_input_tokens=1_000_000)})
        project = cost_sessions([usd, cny], table)
        assert set(project.totals_by_currency.keys()) == {"USD", "CNY"}
        assert project.total_cost is None          # 多币种不假求和
        assert project.delta_vs_self_reported is None
        assert any("汇率" in w for w in project.warnings)
        assert any("币种" in w for w in project.warnings)
    finally:
        custom.unlink()


def test_cost_sessions_可传汇率换算CNY():
    custom = DATA / "custom_prices.json"
    custom.write_text(json.dumps({
        "rows": {"deepseek-v4-flash": {"input": 1.0, "cache_write_5m": 1.0, "cache_write_1h": 1.0,
                                       "cache_read": 0.02, "output": 4.0, "currency": "CNY"}},
    }), encoding="utf-8")
    try:
        table = load_price_table(custom)
        r = SessionReport(session_id="c", self_reported_cost_usd=10.0,
                          per_model={"deepseek-v4-flash": _usage(new_input_tokens=1_000_000)})
        project = cost_sessions([r], table, fx_cny_to_usd=7.2)
        # 1 CNY / 7.2 ≈ 0.1389 USD
        assert project.delta_vs_self_reported == pytest.approx((1.0 / 7.2 - 10.0) / 10.0)
    finally:
        custom.unlink()


def test_峰谷判定():
    assert is_peak_beijing(datetime(2026, 10, 15, 10, 0)) is True   # 周四 10:00
    assert is_peak_beijing(datetime(2026, 10, 15, 13, 0)) is False  # 午休
    assert is_peak_beijing(datetime(2026, 10, 17, 10, 0)) is False  # 周六
    assert is_peak_beijing(datetime(2026, 10, 15, 2, 0)) is False   # 凌晨


def test_峰谷区间计算():
    u = _usage(new_input_tokens=1_000_000, output_tokens=1_000)
    lo, hi = compute_cost_range(u, DEFAULT_PRICES.rows["deepseek-v4-flash"],
                                DEFAULT_PRICES.rows["deepseek-v4-flash-peak"])
    assert lo.total <= hi.total
    assert hi.total == pytest.approx(lo.total * 2)


def test_costs_to_json_输出结构():
    r = parse_session(DATA / "sample_new_only.jsonl")
    out = costs_to_json(cost_sessions([r], DEFAULT_PRICES))
    parsed = json.loads(out)
    assert parsed["totals_by_currency"]["USD"] > 0
    first = parsed["per_session"][0]
    assert first["session_id"] == "sess-1"
    assert first["tiers"]["cache_read"] == pytest.approx(26000 / 1e6 * 0.2)
