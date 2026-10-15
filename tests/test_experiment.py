"""模块 E（experiment）测试：合成任务确定性、缓存模型手算锚点、编排与报告。"""
import csv
import json
from types import SimpleNamespace

import pytest

from ctxautopsy.billing import PriceRow
from ctxautopsy.compressors import create_compressor
from ctxautopsy.experiment.cache_model import (
    CacheAssumptions,
    CompressPlan,
    simulate_turns,
)
from ctxautopsy.experiment.runner import (
    ExperimentConfig,
    StrategySpec,
    run_experiment,
)
from ctxautopsy.experiment.report import render_markdown, write_csv
from ctxautopsy.experiment.synthetic import TaskConfig, generate_task
from ctxautopsy.schema import Msg


# ---------------------------------------------------------------- 合成任务

def test_generate_task_deterministic():
    a = generate_task("t", 7).to_dict()
    b = generate_task("t", 7).to_dict()
    assert a == b


def test_generate_task_structure():
    cfg = TaskConfig(n_turns=4, n_constraints=3, n_negations=4,
                     n_tool_pairs_done=4, n_tool_pairs_inflight=2,
                     n_numeric_facts=4, resident_chars=400,
                     tool_result_max_chars=100, thinking_chars_per_turn=100)
    task = generate_task("t", 7, cfg)
    assert len(task.constraints) == 3
    assert len(task.negations) == 4
    assert len(task.numeric_facts) == 4
    done = [p for p in task.tool_pairs if p.done]
    inflight = [p for p in task.tool_pairs if not p.done]
    assert len(done) == 4 and len(inflight) == 2
    # 在途链：call 在 turns 中出现，result 永不出现
    for p in inflight:
        calls = [m for turn in task.turns for m in turn
                 if m.kind == "tool_call" and m.tool_call_id == p.call_id]
        results = [m for turn in task.turns for m in turn
                   if m.kind == "tool_result" and m.tool_call_id == p.call_id]
        assert calls and not results


# ---------------------------------------------------------------- 缓存模型手算锚点

def _block(uid, ch, n_chars=16):
    """4 token 块（chars_per_token=4）。"""
    return Msg(uid=uid, role="user", kind="text", text=ch * n_chars)


def _anchor_task():
    """设计文档手算样例：resident 4 token、turn1 三块、turn2 两块、turn3 一块。"""
    resident = [_block("R", "R")]
    turns = [
        [_block("A", "A"), _block("B", "B"), _block("C", "C")],
        [_block("D", "D"), _block("E", "E")],
        [_block("U3", "U")],
    ]
    return SimpleNamespace(resident=resident, turns=turns,
                           cfg=SimpleNamespace(n_turns=3))


_ANCHOR_PRICE = PriceRow(input=3.0, cache_write_5m=3.0, cache_write_1h=3.0,
                         cache_read=0.3, output=15.0, source="测试锚点价")
_ANCHOR_CACHE = CacheAssumptions(chars_per_token=4.0, breakpoint_tokens=4,
                                 min_cache_tokens=4)


def _slide8(history, params):
    c = create_compressor("sliding_window", {"max_tokens": 8})
    return list(c.compress(history, params).messages)


def test_cache_baseline_hand_calc():
    """手算锚点：baseline turn3 应为 cr=20 / cw=8 / ni=0，成本 30（20×0.3 + 8×3）。"""
    turns = simulate_turns(_anchor_task(), CompressPlan.no_compression(),
                           _ANCHOR_CACHE, _ANCHOR_PRICE, None)
    t3 = turns[2]
    assert t3.prompt_tokens == 28
    assert (t3.tiers.cache_read, t3.tiers.cache_write, t3.tiers.new_input) == (20, 8, 0)
    assert t3.cost_usd == pytest.approx((20 * 0.3 + 8 * 3) / 1e6)


def test_cache_compressed_hand_calc():
    """手算锚点：every_turn 滑窗（保最后 2 块）turn3 应为 cr=4 / cw=12 / ni=0，
    成本 37.2 —— 历史压掉 60%，当轮账单反而 +24%（省 token 反涨价的结构复现）。"""
    plan = CompressPlan(compress_from_turn=2, every_turn=True,
                        strategy="sliding_window", params={"max_tokens": 8})
    turns = simulate_turns(_anchor_task(), plan, _ANCHOR_CACHE, _ANCHOR_PRICE, _slide8)
    t3 = turns[2]
    assert t3.prompt_tokens == 16
    assert (t3.tiers.cache_read, t3.tiers.cache_write, t3.tiers.new_input) == (4, 12, 0)
    assert t3.cost_usd == pytest.approx((4 * 0.3 + 12 * 3) / 1e6)
    # 省 token 反涨价：与 baseline turn3 比，prompt 少 12 token 但账单涨 24%
    base_t3 = simulate_turns(_anchor_task(), CompressPlan.no_compression(),
                             _ANCHOR_CACHE, _ANCHOR_PRICE, None)[2]
    assert t3.cost_usd > base_t3.cost_usd


def test_cache_multi_breakpoint_optimistic_le_pessimistic():
    task = _anchor_task()
    pess = CacheAssumptions(chars_per_token=4.0, breakpoint_tokens=4,
                            min_cache_tokens=4, multi_breakpoint=False)
    opti = CacheAssumptions(chars_per_token=4.0, breakpoint_tokens=4,
                            min_cache_tokens=4, multi_breakpoint=True)
    plan = CompressPlan(compress_from_turn=2, strategy="sliding_window",
                        params={"max_tokens": 8})
    p_turns = simulate_turns(task, plan, pess, _ANCHOR_PRICE, _slide8)
    o_turns = simulate_turns(task, plan, opti, _ANCHOR_PRICE, _slide8)
    assert sum(t.cost_usd for t in o_turns) <= sum(t.cost_usd for t in p_turns) + 1e-9


# ---------------------------------------------------------------- 编排与报告

def _small_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        tasks=["t_short"],
        strategies=[StrategySpec("baseline", {}),
                    StrategySpec("truncate_head", {"keep_ratio": 0.5}),
                    StrategySpec("freq_drop", {"mode": "stopwords", "top_ratio": 0.05})],
        repeats=2,
        compress_from_turn=3,
        cache=CacheAssumptions(chars_per_token=4.0, breakpoint_tokens=64,
                               min_cache_tokens=64),
        price_model="deepseek-v4-flash",
    )


def test_run_experiment_row_count_and_schema():
    from ctxautopsy.experiment.synthetic import TaskConfig as TC
    cfg = _small_cfg()
    result = run_experiment(cfg, task_factory=lambda tid, seed: generate_task(
        tid, seed, TC(n_turns=4, n_constraints=3, n_negations=4,
                      n_tool_pairs_done=4, n_tool_pairs_inflight=2,
                      n_numeric_facts=4, resident_chars=2000,
                      tool_result_max_chars=200, thinking_chars_per_turn=100)))
    assert len(result.rows) == 1 * 3 * 2  # 任务 × 策略 × 重复
    for r in result.rows:
        assert r.tokens_before >= 0 and r.compression_rate_pct >= 0
        assert all(v >= 0 for v in (r.damage_constraint, r.damage_negation,
                                    r.damage_toolchain, r.damage_numeric))


def test_run_experiment_baseline_cell_zero():
    from ctxautopsy.experiment.synthetic import TaskConfig as TC
    cfg = _small_cfg()
    result = run_experiment(cfg, task_factory=lambda tid, seed: generate_task(
        tid, seed, TC(n_turns=4, resident_chars=2000, tool_result_max_chars=200,
                      thinking_chars_per_turn=100)))
    for r in result.rows:
        if r.strategy == "baseline":
            assert r.compression_rate_pct == 0
            assert r.damage_constraint == 0 and r.damage_negation == 0
            assert r.damage_toolchain == 0 and r.damage_numeric == 0
            assert r.cost_delta_usd == 0


def test_run_experiment_deterministic():
    from ctxautopsy.experiment.synthetic import TaskConfig as TC
    cfg = _small_cfg()
    f = lambda tid, seed: generate_task(tid, seed, TC(n_turns=4, resident_chars=2000,
                                                     tool_result_max_chars=200,
                                                     thinking_chars_per_turn=100))
    a = run_experiment(cfg, task_factory=f)
    b = run_experiment(cfg, task_factory=f)
    assert a.to_csv_rows() == b.to_csv_rows()


def test_run_experiment_rejects_missing_baseline():
    cfg = _small_cfg()
    cfg.strategies = [StrategySpec("truncate_head", {})]
    with pytest.raises(ValueError, match="baseline"):
        run_experiment(cfg)


def test_write_csv_roundtrip(tmp_path):
    from ctxautopsy.experiment.synthetic import TaskConfig as TC
    cfg = _small_cfg()
    result = run_experiment(cfg, task_factory=lambda tid, seed: generate_task(
        tid, seed, TC(n_turns=4, resident_chars=2000, tool_result_max_chars=200,
                      thinking_chars_per_turn=100)))
    p = write_csv(result, str(tmp_path / "metrics.csv"))
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(result.rows)
    assert rows[0]["task_id"] == "t_short"


def test_render_markdown_has_honesty_sections(tmp_path):
    from ctxautopsy.experiment.synthetic import TaskConfig as TC
    from ctxautopsy.billing import DEFAULT_PRICES
    cfg = _small_cfg()
    result = run_experiment(cfg, task_factory=lambda tid, seed: generate_task(
        tid, seed, TC(n_turns=4, resident_chars=2000, tool_result_max_chars=200,
                      thinking_chars_per_turn=100)))
    md = render_markdown(result, DEFAULT_PRICES.rows["deepseek-v4-flash"])
    assert "缓存建模假设清单" in md
    assert "价格表与来源" in md
    assert "复现方法" in md
    assert "估算" in md  # 口径标注
