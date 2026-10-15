"""模块 C（compressors）测试：五个注册策略 + 不变式。"""
import json

import pytest

from ctxautopsy.compressors import create_compressor, list_compressors
from ctxautopsy.schema import Msg


def _msg(uid, kind="text", text="", role="user", tool_call_id=None, tool_name=None, tokens=None):
    return Msg(uid=uid, role=role, kind=kind, text=text,
               tool_call_id=tool_call_id, tool_name=tool_name, tokens=tokens)


@pytest.fixture
def five():
    """5 条消息：约束 + 工具链 + 长文本。"""
    return [
        _msg("u1", text="【约束】不要删除这个文件，温度不能超过 75.5 度"),
        _msg("u2", kind="tool_call", text='{"pattern": "**/*.py"}',
             tool_call_id="call_1", tool_name="Glob", role="assistant"),
        _msg("u3", kind="tool_result", text="foo.py\nbar.py", tool_call_id="call_1"),
        _msg("u4", text="今天天气很好我们出去玩一共花了二百三十块钱" * 10),
        _msg("u5", text="结尾备注：本会话由自动化脚本生成。"),
    ]


# ---- 注册表与工厂 ----

def test_registry_and_factory():
    names = list_compressors()
    assert names == ["freq_drop", "rule_summary", "sliding_window",
                     "truncate_head", "truncate_tail"]
    for n in names:
        c = create_compressor(n)
        assert c.name == n
    with pytest.raises(KeyError):
        create_compressor("nope")


# ---- 截断 ----

def test_truncate_head_keeps_first_n(five):
    r = create_compressor("truncate_head", {"keep_items": 2}).compress(five)
    assert [m.uid for m in r.messages] == ["u1", "u2"]
    assert len(r.changes) == 3
    assert all(c.op == "delete" and c.reason == "truncate_head" for c in r.changes)
    assert all(c.orig_text and c.new_text is None for c in r.changes)


def test_truncate_tail_keeps_last_n(five):
    r = create_compressor("truncate_tail", {"keep_items": 2}).compress(five)
    assert [m.uid for m in r.messages] == ["u4", "u5"]


def test_truncate_keep_ratio_uses_ceil():
    msgs = [_msg(f"u{i}", text="abcd") for i in range(3)]
    r = create_compressor("truncate_head", {"keep_ratio": 0.5}).compress(msgs)
    assert len(r.messages) == 2  # ceil(1.5)


def test_truncate_more_than_length_no_changes(five):
    r = create_compressor("truncate_head", {"keep_items": 10}).compress(five)
    assert r.messages == five
    assert r.changes == []


def test_truncate_zero_keep(five):
    r = create_compressor("truncate_head", {"keep_items": 0}).compress(five)
    assert r.messages == []
    assert r.stats.saved_ratio == 1.0


# ---- 滑窗 ----

def test_sliding_window_respects_budget():
    msgs = [_msg(f"u{i}", text="a" * 40, tokens=10) for i in range(5)]  # 每条 10 token
    r = create_compressor("sliding_window", {"max_tokens": 25}).compress(msgs)
    assert sum(m.tokens or 10 for m in r.messages) <= 25
    # 保留尾部 2 条（20 token），第 3 条从尾部算起被掐头保留 5 token（"a"*20）
    assert [m.uid for m in r.messages] == ["u2", "u3", "u4"]
    assert r.messages[0].text == "a" * 20
    assert r.messages[0].tokens == 5  # 掐头后 tokens 字段同步为估算值
    trim = [c for c in r.changes if c.op == "modify"]
    assert len(trim) == 1 and trim[0].reason == "window_trim"


def test_sliding_window_cut_inside_tool_call():
    msgs = [
        _msg("u1", kind="tool_call", text='{"pattern": "**/*.py"}', tokens=6,
             tool_call_id="call_1", tool_name="Glob", role="assistant"),
        _msg("u2", kind="tool_result", text="x" * 80, tokens=20, tool_call_id="call_1"),
    ]
    r = create_compressor("sliding_window", {"max_tokens": 20}).compress(msgs)
    # 预算 20：tool_result(20) 全保留，tool_call 无预算 → 掐头保留 0 字符 → 升级 delete
    assert [m.uid for m in r.messages] == ["u2"]
    assert r.changes[0].op == "delete"


# ---- 按词频删词 ----

def test_freq_drop_stopwords_removes_negation():
    text = "【约束】不要删除这段内容。不要删除，不要删除，不要删除。"
    msgs = [_msg("u1", text=text)]
    r = create_compressor("freq_drop", {"mode": "stopwords", "top_ratio": 0.5}).compress(msgs)
    c = r.changes[0]
    assert c.op == "modify" and c.reason == "freq_drop_stopwords"
    assert "不" in c.meta["dropped_tokens"] or "要" in c.meta["dropped_tokens"]
    assert "不要删除" not in c.new_text


def test_freq_drop_default_skips_tool_kinds(five):
    r = create_compressor("freq_drop", {"mode": "stopwords", "top_ratio": 0.1}).compress(five)
    assert not any(c.kind != "text" for c in r.changes)


def test_freq_drop_empty_after_drop_becomes_delete():
    msgs = [_msg("u1", text="的的的的的")]
    r = create_compressor("freq_drop", {"mode": "stopwords", "top_ratio": 1.0}).compress(msgs)
    assert r.messages == []
    assert r.changes[0].op == "delete"


def test_freq_drop_rare_mode_drops_numbers():
    msgs = [_msg("u1", text="常量 pi = 3.14，用于计算面积。面积公式反复使用反复使用。")]
    r = create_compressor("freq_drop", {"mode": "rare", "top_ratio": 0.1}).compress(msgs)
    assert any("3.14" in c.meta.get("dropped_tokens", []) for c in r.changes)


def test_freq_drop_tie_break_deterministic():
    text = "的的不不的的不不"  # 的×4 不×4 平手
    msgs = [_msg("u1", text=text)]
    c = create_compressor("freq_drop", {"mode": "stopwords", "top_ratio": 0.25})
    r1 = c.compress(msgs).to_dict()
    r2 = c.compress(msgs).to_dict()
    assert r1 == r2


# ---- 规则摘要 ----

def test_rule_summary_long_message_summarized():
    long_text = "第一句话说明了背景和动机。第二句话包含大量细节：" + "细节" * 300
    msgs = [_msg("u1", text=long_text)]
    r = create_compressor("rule_summary", {"min_chars": 100, "summary_len": 80}).compress(msgs)
    c = r.changes[0]
    assert c.op == "modify" and c.reason == "rule_summary"
    assert c.new_text.startswith("第一句话说明了背景和动机。")
    assert f"原{len(long_text)}字" in c.new_text


def test_rule_summary_short_message_untouched():
    msgs = [_msg("u1", text="短消息")]
    r = create_compressor("rule_summary", {"min_chars": 100}).compress(msgs)
    assert r.changes == []


def test_rule_summary_approx_numbers():
    msgs = [_msg("u1", text="价格为 1234.567 元，共 0.05 克。" + "。".join(["长"] * 100))]
    r = create_compressor("rule_summary",
                          {"min_chars": 50, "approx_numbers": True}).compress(msgs)
    c = r.changes[0]
    assert "约1200" in c.new_text or "约1234" in c.new_text
    assert "约0.05" in c.new_text


# ---- 不变式与确定性 ----

def test_all_strategies_deterministic(five):
    for name in list_compressors():
        c = create_compressor(name, {"max_tokens": 30} if name == "sliding_window" else None)
        assert c.compress(five).to_dict() == c.compress(five).to_dict(), name


def test_json_round_trip_with_chinese(five):
    r = create_compressor("truncate_head", {"keep_items": 3}).compress(five)
    from ctxautopsy.schema import CompressResult
    back = CompressResult.from_dict(json.loads(r.to_json()))
    assert back.to_dict() == r.to_dict()


def test_invalid_uid_raises():
    with pytest.raises(ValueError, match="uid"):
        create_compressor("truncate_head").compress([_msg("", text="x"), _msg("u2", text="y")])
    with pytest.raises(ValueError, match="uid"):
        create_compressor("truncate_head").compress([_msg("u1", text="x"), _msg("u1", text="y")])


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="keep_item"):
        create_compressor("truncate_head").compress(
            [_msg("u1", text="x")], {"keep_item": 5})


def test_output_is_subsequence(five):
    for name in list_compressors():
        c = create_compressor(name, {"max_tokens": 30} if name == "sliding_window" else None)
        r = c.compress(five)
        in_uids = [m.uid for m in five]
        out_uids = [m.uid for m in r.messages]
        it = iter(in_uids)
        assert all(uid in it for uid in out_uids), name


def test_constraint_marker_not_special():
    """压缩器不保护约束：含【约束】的消息与普通文本同等对待。"""
    msgs = [_msg("u1", text="【约束】不要不要不要删除内容" * 3)]
    r = create_compressor("freq_drop", {"mode": "stopwords", "top_ratio": 0.2}).compress(msgs)
    assert len(r.changes) >= 1
