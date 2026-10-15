# -*- coding: utf-8 -*-
"""模块 A（session_log）的 pytest 测试 = 最终接口契约。

fixture 结构取自本机 ~/.claude/projects/ 实测 jsonl 的行形态：
- assistant 条目 message.usage 含 input_tokens / cache_creation_input_tokens /
  cache_read_input_tokens / output_tokens，另有 cache_creation 分档对象；
- 本机网关语义：input_tokens 不含缓存读（95% 调用 input < cache_read）；
- 噪音条目：queue-operation / summary / progress / attachment / cost-state 等。
"""
import json
from pathlib import Path

import pytest

from ctxautopsy.session_log import (
    aggregate_sessions,
    parse_session,
    parse_sessions_dir,
    sessions_to_json,
)

DATA = Path(__file__).parent / "data"


def test_parse_session_四档用量_new_only语义():
    """new_only 语义（input 不含缓存读）：input 全部视为新输入。"""
    report = parse_session(DATA / "sample_new_only.jsonl")
    assert report.usage_semantics == "new_only"
    u = report.usage
    assert u.new_input_tokens == 25150
    assert u.cache_write_tokens == 1500
    assert u.cache_write_5m_tokens == 1000
    assert u.cache_write_1h_tokens == 500
    assert u.cache_read_tokens == 26000
    assert u.output_tokens == 300
    assert u.api_calls == 2
    assert u.input_tokens_total == 52650  # 新输入+缓存写+缓存读


def test_parse_session_total语义扣减():
    """total 语义（input 是总量）：新输入 = input − 缓存写 − 缓存读。"""
    report = parse_session(DATA / "sample_total_semantics.jsonl")
    assert report.usage_semantics == "total"
    u = report.usage
    assert u.new_input_tokens == 500       # 28000 - 26000 - 1500
    assert u.cache_write_tokens == 1500    # 无分档对象时回退自 cache_creation_input_tokens
    assert u.cache_read_tokens == 26000
    assert u.output_tokens == 50
    assert report.api_calls == 1


def test_parse_session_旧格式缺缓存字段按0计():
    """只有 input/output 的旧版 usage：缓存两档为 0，不丢调用、不抛异常。"""
    report = parse_session(DATA / "sample_old_format.jsonl")
    u = report.usage
    assert u.new_input_tokens == 100
    assert u.cache_write_tokens == 0
    assert u.cache_read_tokens == 0
    assert u.output_tokens == 50
    assert report.api_calls == 1
    assert report.usage_semantics == "total"
    assert report.warnings == []  # 无缓存读样本时不告警


def test_parse_session_噪音行与坏行计数():
    report = parse_session(DATA / "sample_new_only.jsonl")
    assert report.total_lines == 12
    assert report.parsed_lines == 11
    assert report.parse_errors == 1
    assert report.skipped_by_type["queue-operation"] == 1
    assert report.skipped_by_type["summary"] == 1
    assert report.skipped_by_type["progress"] == 1
    assert report.skipped_by_type["attachment"] == 1
    assert report.skipped_by_type["cost-state"] == 1


def test_常驻开销_new_only首条调用扣用户文本():
    report = parse_session(DATA / "sample_new_only.jsonl")
    # 首条调用 input=25000；用户文本 9 CJK + 1 全角逗号 ≈ 9（CJK≈1 token/字）
    assert report.composition.resident_overhead_tokens == 24991
    assert report.composition.overhead_method == "first_call_minus_user_text"


def test_常驻开销_total语义扣除缓存档():
    report = parse_session(DATA / "sample_total_semantics.jsonl")
    # 28000 − 缓存读 26000 − 缓存写 1500 − 用户文本 "hello world" ≈ 3 = 497
    assert report.composition.resident_overhead_tokens == 497
    assert report.composition.overhead_method == "first_call_minus_cache_minus_user_text"


def test_会话构成桶统计():
    report = parse_session(DATA / "sample_new_only.jsonl")
    c = report.composition
    assert c.user_text_tokens == 9          # 你好，帮我写一个函数
    assert c.tool_output_tokens == 4        # "result text here" 16 ASCII ≈ 4
    assert c.thinking_tokens == 6           # "let me think about this" 23 ASCII ≈ 6
    assert c.compact_summary_tokens == 15   # 60 ASCII ≈ 15
    assert c.tool_result_blocks == 1
    assert c.thinking_blocks == 1


def test_isMeta行不计入用户文本():
    report = parse_session(DATA / "sample_new_only.jsonl")
    # meta 行（str 内容）与 tool_result 行都未计入 user_text
    assert report.composition.user_text_tokens == 9


def test_per_model分模型统计():
    report = parse_session(DATA / "sample_new_only.jsonl")
    m = report.per_model["claude-sonnet-5-5"]
    assert m.new_input_tokens == 25150
    assert m.cache_read_tokens == 26000
    assert m.output_tokens == 300


def test_harness自报成本读取():
    report = parse_session(DATA / "sample_new_only.jsonl")
    assert report.self_reported_cost_usd == pytest.approx(0.123)


def test_空文件返回全零报告():
    empty = DATA / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    try:
        report = parse_session(empty)
        assert report.api_calls == 0
        assert report.usage_semantics == "none"
        assert report.composition.overhead_method == ""
        assert report.warnings  # 有“无 usage 数据”告警
    finally:
        empty.unlink()


def test_json输出不包含任何消息文本():
    """隐私硬约束：序列化结果只含聚合数字，哨兵文本不得出现。"""
    report = parse_session(DATA / "sample_new_only.jsonl")
    out = sessions_to_json([report])
    for sentinel in ("帮我写一个函数", "result text here", "let me think about this",
                     "This session is being continued", "TOP_SECRET"):
        assert sentinel not in out, f"隐私泄漏: {sentinel}"
    parsed = json.loads(out)
    assert parsed[0]["session_id"] == "sess-1"
    assert parsed[0]["usage"]["new_input_tokens"] == 25150
    assert parsed[0]["composition"]["resident_overhead_tokens"] == 24991
    assert parsed[0]["per_model"]["claude-sonnet-5-5"]["output_tokens"] == 300


def test_parse_sessions_dir_只取jsonl且按文件名排序():
    reports = parse_sessions_dir(DATA)
    names = [r.source_file for r in reports]
    assert names == sorted(names)
    assert all(n.endswith(".jsonl") for n in names)
    assert "sample_new_only.jsonl" in names


def test_parse_sessions_dir_limit限制文件数():
    reports = parse_sessions_dir(DATA, limit=1)
    assert len(reports) == 1


def test_aggregate_sessions_汇总正确():
    r1 = parse_session(DATA / "sample_new_only.jsonl")
    r2 = parse_session(DATA / "sample_total_semantics.jsonl")
    summary = aggregate_sessions([r1, r2])
    assert summary.sessions == 2
    assert summary.totals.api_calls == 3
    assert summary.totals.new_input_tokens == 25150 + 500
    assert summary.total_self_reported_cost_usd == pytest.approx(0.123)
    assert "无 harness 自报成本" in summary.warnings[0]  # 第二个会话没有 cost-state


def test_真机数据冒烟():
    """对 ~/.claude/projects 真实日志跑一遍：产出非零统计、语义为实测的 new_only。"""
    real = Path.home() / ".claude/projects/c--Users-spz97-Desktop-VScode/066ce6ce-db2f-4704-812a-603d732c2291.jsonl"
    if not real.exists():
        pytest.skip("本机无真实会话日志")
    report = parse_session(real)
    assert report.usage_semantics == "new_only"
    assert report.api_calls > 100
    assert report.usage.cache_read_tokens > 0
    assert report.usage.new_input_tokens > 0
    assert report.composition.resident_overhead_tokens > 0
    assert report.self_reported_cost_usd is not None
