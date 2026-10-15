"""模块 D（checkers）测试：四项检查的判定与假阳性防线。"""
import json

from ctxautopsy.checkers import (
    CheckerConfig,
    CompressedMessage,
    Constraint,
    MessageText,
    NegationSpec,
    ToolCall,
    check_constraints,
    check_negations,
    check_numbers,
    check_toolchain,
    run_all_checks,
)


def _orig(*texts):
    return [MessageText(message_id=f"syn-{i}", role="user", text=t) for i, t in enumerate(texts)]


def _comp(*texts):
    return [CompressedMessage(message_id=f"syn-{i}", text=t) for i, t in enumerate(texts)]


# ---- 否定词 ----

def test_negation_flip():
    r = check_negations(_orig("不需要登录"), _comp("需要登录"))
    assert r.hit_count == 1
    h = r.hits[0]
    assert h.kind == "polarity_flip"
    assert h.snippet == "不需要"
    assert h.extra["head"] == "需要"


def test_negation_whole_sentence_deleted_no_hit():
    r = check_negations(_orig("不需要登录"), _comp(""))
    assert r.hit_count == 0
    assert r.stats["negation_lost"] == 1


def test_negation_preserved_no_hit():
    r = check_negations(_orig("这个字段不需要校验"), _comp("这个字段不需要校验"))
    assert r.hit_count == 0


def test_negation_head_count_conservation():
    r = check_negations(_orig("不需要A，需要B"), _comp("需要B"))
    assert r.hit_count == 0


def test_negation_stopphrase():
    r = check_negations(_orig("差不多就行"), _comp("差不多就行"))
    assert r.hit_count == 0


def test_negation_english():
    r = check_negations(_orig("not delete the file"), _comp("delete the file"))
    assert r.hit_count == 1
    assert r.hits[0].kind == "polarity_flip"


def test_negation_config_override():
    cfg = CheckerConfig(negation_specs=[NegationSpec("严禁", 2)])
    r = check_negations(_orig("严禁写入缓存"), _comp("写入缓存"), config=cfg)
    assert r.hit_count == 1


# ---- 约束 ----

def test_constraint_three_states():
    constraints = [Constraint("禁止删除数据库"), Constraint("保留原始文件名"), Constraint("每天备份一次")]
    r = check_constraints(
        _orig("【约束】禁止删除数据库", "【约束】保留原始文件名", "【约束】每天备份一次"),
        _comp("删除数据库", "保留原始文件名"),
        constraints,
    )
    assert r.stats == {"total_constraints": 3, "preserved": 1, "fragmented": 1, "lost": 1}
    kinds = {h.kind for h in r.hits}
    assert kinds == {"constraint_fragmented", "constraint_lost"}


def test_constraint_normalization():
    r = check_constraints(
        _orig("Do NOT delete"), _comp("do not delete"), [Constraint("Do NOT delete")])
    assert r.hit_count == 0


# ---- 工具链 ----

def test_toolchain_call_without_result():
    orig = [MessageText(message_id="m1", role="assistant", text="ls",
                        tool_calls=[ToolCall("call_00_abc", "Bash")]),
            MessageText(message_id="m2", role="user", text="result",
                        tool_result_ids=["call_00_abc"])]
    comp = [CompressedMessage(message_id="m1", text="ls", kind="tool_call")]
    r = check_toolchain(orig, comp)
    assert r.hit_count == 1
    assert r.hits[0].kind == "call_without_result"
    assert r.hits[0].extra["tool_name"] == "Bash"


def test_toolchain_orphan_and_removed():
    orig = [MessageText(message_id="m1", role="assistant", text="ls",
                        tool_calls=[ToolCall("call_1", "Bash")]),
            MessageText(message_id="m2", role="user", text="result",
                        tool_result_ids=["call_1"])]
    # result 在 call 不在 → orphan
    r = check_toolchain(orig, [CompressedMessage(message_id="m2", text="result",
                                                 kind="tool_result")])
    assert r.hit_count == 1 and r.hits[0].kind == "orphan_result"
    # 两者都不在 → removed_pairs、0 命中
    r2 = check_toolchain(orig, [])
    assert r2.hit_count == 0
    assert r2.stats["removed_pairs"] == 1
    assert r2.stats["intact_pairs"] == 0


# ---- 数值 ----

def test_numbers_approx_lost():
    r = check_numbers(_orig("3.14159265"), _comp("3.14"))
    assert r.hit_count == 1 and r.hits[0].kind == "approx"
    r2 = check_numbers(_orig("1024"), _comp(""))
    assert r2.hit_count == 1 and r2.hits[0].kind == "lost"
    r3 = check_numbers(_orig("3.14159265"), _comp("约 3.1"))
    assert r3.hit_count == 1 and r3.hits[0].kind == "approx_word"


def test_numbers_value_equal():
    r = check_numbers(_orig("66% 62K 1,024"), _comp("0.66 62000 1024"))
    assert r.hit_count == 0
    assert r.stats["preserved"] == 3


def test_numbers_date_excluded():
    r = check_numbers(_orig("今天日期 2026-10-15"), _comp("今天日期"))
    assert r.stats["total_numbers"] == 0


def test_numbers_multiset_greedy():
    r = check_numbers(_orig("100 和 100"), _comp("100"))
    assert r.stats == {"total_numbers": 2, "preserved": 1, "approx": 0,
                       "approx_word": 0, "lost": 1}


# ---- 汇总与边界 ----

def test_identity_baseline():
    original = _orig("不需要登录", "3.14")
    comp = _comp("不需要登录", "3.14")
    report = run_all_checks(original, comp, [Constraint("不需要登录")])
    assert report.total_hits == 0


def test_empty_compressed():
    original = _orig("【约束】禁止删除数据库", "3.14")
    report = run_all_checks(original, _comp(), [Constraint("禁止删除数据库")])
    kinds = {h.kind for r in report.results for h in r.hits}
    assert "constraint_lost" in kinds
    assert any(r.stats["lost"] == 1 for r in report.results if r.checker == "numeric")
    neg = [r for r in report.results if r.checker == "negation"][0]
    assert neg.hit_count == 0  # 翻转判据：整段丢失不算否定词删除


def test_determinism():
    original = _orig("【约束】禁止删除数据库，温度 75.5 度。不需要登录。")
    comp = _comp("删除数据库，温度 75 度。需要登录。")
    a = run_all_checks(original, comp, [Constraint("禁止删除数据库")]).to_dict()
    b = run_all_checks(original, comp, [Constraint("禁止删除数据库")]).to_dict()
    assert a == b


def test_json_roundtrip():
    original = _orig("【约束】禁止删除数据库")
    report = run_all_checks(original, _comp("删除数据库"), [Constraint("禁止删除数据库")])
    text = report.to_json()
    assert "\\u" not in text  # ensure_ascii=False
    d = json.loads(text)
    assert d["results"][0]["checker"] == "constraint"
    assert d["results"][0]["hits"][0]["kind"] == "constraint_fragmented"
