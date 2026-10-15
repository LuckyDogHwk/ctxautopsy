"""四项确定性检查器（模块 D）：压缩前后的"验尸"。

输入：压缩前的原始会话视图 + 压缩后的产物，输出四项损坏检查的 JSON 报告。
全部为纯字符串/数值算法：无随机、无 LLM 当裁判，同输入同输出有唯一正确答案。

1. 约束保全（文本级）：注入的硬约束字符串在压缩后是否仍完整存在
   （PRESERVED / FRAGMENTED / LOST 三态，n-gram 重叠率判定）
2. 否定词删除：只有「极性翻转」才算命中——否定词被删而谓词头仍存活
   （假阳性防线：整段丢失、正常表述、固定搭配「差不多」都不误报）
3. 在途工具链截断：tool_call 与 tool_result 配对（靠结构字段，不靠 id 正则）
4. 数值变形：精确数字在压缩后消失 / 变成近似值 / 被「约 X」约化

分工：本模块只读两个视图（原始 / 压缩），不读 jsonl、不读磁盘；
真实日志的结构级检查（工具链）与合成实验（约束/否定词/数值，有 ground truth）
走同一套接口。
"""
from dataclasses import dataclass, field
import json
import re

from .schema import DEFAULT_NUMBER_PATTERN, Msg, is_cjk


# ---------------------------------------------------------------- 输入视图

@dataclass
class ToolCall:
    tool_use_id: str
    name: str


@dataclass
class MessageText:
    """原始侧消息文本化视图。"""
    message_id: str
    role: str
    text: str
    tool_calls: list = field(default_factory=list)      # assistant 消息的 tool_use 块
    tool_result_ids: list = field(default_factory=list)  # user 消息的 tool_result 配对 id


@dataclass
class CompressedMessage:
    """压缩侧视图。kind 用于工具链的结构化判定（C 的契约保证 uid/kind 永不受损）。"""
    message_id: str | None
    text: str
    kind: str | None = None


@dataclass
class Constraint:
    text: str
    source_message_id: str | None = None


@dataclass
class NegationSpec:
    token: str
    scope_head_len: int          # 谓词头长度：CJK 取 2 字，英文取 1 词


DEFAULT_NEGATIONS = [
    NegationSpec("不要", 2), NegationSpec("不得", 2), NegationSpec("不能", 2),
    NegationSpec("不应", 2), NegationSpec("不可", 2), NegationSpec("不准", 2),
    NegationSpec("无需", 2), NegationSpec("禁止", 2), NegationSpec("切勿", 2),
    NegationSpec("避免", 2), NegationSpec("拒绝", 2),
    NegationSpec("不", 2), NegationSpec("勿", 2), NegationSpec("别", 2),
    NegationSpec("do not", 1), NegationSpec("must not", 1), NegationSpec("cannot", 1),
    NegationSpec("shouldn't", 1), NegationSpec("don't", 1), NegationSpec("can't", 1),
    NegationSpec("never", 1), NegationSpec("without", 1), NegationSpec("prohibited", 1),
    NegationSpec("forbidden", 1), NegationSpec("not", 1), NegationSpec("no", 1),
]

# 「不」固定搭配停用表：其中的否定字不构成谓词否定，跳过
DEFAULT_NEGATION_STOPPHRASES = {
    "差不多", "不断", "不仅", "不过", "不错", "不得不", "不得已", "恨不得",
    "不及", "不计", "不停", "不止", "大不了", "了不起",
}

DEFAULT_NUMBER_PATTERN = DEFAULT_NUMBER_PATTERN  # 从 schema 导入，此处保持模块内可用


# ---------------------------------------------------------------- 配置与输出

@dataclass
class CheckerConfig:
    constraint_overlap_threshold: float = 0.5
    negation_specs: list = field(default_factory=lambda: list(DEFAULT_NEGATIONS))
    negation_stopphrases: set = field(default_factory=lambda: set(DEFAULT_NEGATION_STOPPHRASES))
    report_lost_negations: bool = False
    number_approx_band: float = 0.10
    number_pattern: str = DEFAULT_NUMBER_PATTERN
    exclude_patterns: tuple = (r"\d{4}-\d{1,2}-\d{1,2}",)   # 日期等排除形态
    approx_markers: tuple = ("约", "大约", "大概", "近似", "≈", "~", "approx", "around")

    def to_dict(self) -> dict:
        return {
            "constraint_overlap_threshold": self.constraint_overlap_threshold,
            "number_approx_band": self.number_approx_band,
            "negation_specs": [{"token": s.token, "scope_head_len": s.scope_head_len}
                               for s in self.negation_specs],
            "negation_stopphrases": sorted(self.negation_stopphrases),
            "report_lost_negations": self.report_lost_negations,
            "approx_markers": list(self.approx_markers),
        }


@dataclass
class Hit:
    checker: str
    kind: str
    message_id: str | None
    message_index: int | None
    offset_start: int
    offset_end: int
    snippet: str
    detail: str
    compressed_message_id: str | None = None
    compressed_message_index: int | None = None
    compressed_offset_start: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checker": self.checker, "kind": self.kind,
            "message_id": self.message_id, "message_index": self.message_index,
            "offset_start": self.offset_start, "offset_end": self.offset_end,
            "snippet": self.snippet, "detail": self.detail,
            "compressed_message_id": self.compressed_message_id,
            "compressed_message_index": self.compressed_message_index,
            "compressed_offset_start": self.compressed_offset_start,
            "extra": self.extra,
        }


@dataclass
class CheckResult:
    checker: str
    hit_count: int
    hits: list
    stats: dict

    def to_dict(self) -> dict:
        return {"checker": self.checker, "hit_count": self.hit_count,
                "hits": [h.to_dict() for h in self.hits], "stats": self.stats}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class CheckReport:
    results: list          # 恒为 4 项：constraint / negation / toolchain / numeric
    config: dict

    @property
    def total_hits(self) -> int:
        return sum(r.hit_count for r in self.results)

    def to_dict(self) -> dict:
        return {"results": [r.to_dict() for r in self.results], "config": self.config}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------- 工具函数

def flatten_messages(compressed: list[CompressedMessage]) -> str:
    """压缩侧结构化消息 → 单一文本。"""
    return "\n".join(m.text for m in compressed)


def msgs_to_message_texts(msgs: list[Msg]) -> list[MessageText]:
    """模块 C 的 Msg 序列 → 模块 D 的原始侧视图（每条消息一个 MessageText）。"""
    out = []
    for m in msgs:
        mt = MessageText(message_id=m.uid, role=m.role, text=m.text)
        if m.kind == "tool_call":
            mt.tool_calls.append(ToolCall(tool_use_id=m.tool_call_id, name=m.tool_name or ""))
        elif m.kind == "tool_result" and m.tool_call_id:
            mt.tool_result_ids.append(m.tool_call_id)
        out.append(mt)
    return out


def msgs_to_compressed(msgs: list[Msg]) -> list[CompressedMessage]:
    return [CompressedMessage(message_id=m.uid, text=m.text, kind=m.kind) for m in msgs]


def _normalize(text: str) -> str:
    """去全部空白 + 英文统一小写。"""
    return re.sub(r"\s+", "", text).lower()


def _norm_index_map(text: str) -> list[int]:
    """规范化位置 → 原文偏移的映射（只处理去空白；lower() 视作长度不变）。"""
    lowered = text.lower()
    mapping = []
    for raw_pos, ch in enumerate(lowered):
        if not ch.isspace():
            mapping.append(raw_pos)
    return mapping


def _find_in_messages(text: str, needle: str, messages: list[MessageText],
                      normalized: bool = False) -> tuple[int | None, int, int]:
    """在原始侧消息里找 needle（默认原文精确匹配，normalized=True 时规范化匹配）。

    返回 (message_index, offset_start, offset_end)，找不到返回 (None, -1, -1)。
    """
    for idx, m in enumerate(messages):
        hay = _normalize(m.text) if normalized else m.text
        needle_n = _normalize(needle) if normalized else needle
        pos = hay.find(needle_n)
        if pos >= 0:
            if normalized:
                m2 = _norm_index_map(m.text)
                if pos < len(m2):
                    return idx, m2[pos], m2[pos + len(needle_n) - 1] + 1
            return idx, pos, pos + len(needle_n)
    return None, -1, -1


def _locate_in_compressed(text_blob: str, needle: str,
                          compressed: list[CompressedMessage]) -> tuple[int | None, int | None]:
    """在压缩侧定位片段：返回 (message_index, message 内偏移)。"""
    pos = 0
    for idx, m in enumerate(compressed):
        p = m.text.find(needle)
        if p >= 0:
            return idx, p
        pos += len(m.text) + 1
    return None, None


# ---------------------------------------------------------------- 1. 约束保全

def _ngrams(s: str, n: int = 2) -> set:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def check_constraints(original: list[MessageText], compressed: list[CompressedMessage],
                      constraints: list[Constraint],
                      config: CheckerConfig | None = None) -> CheckResult:
    cfg = config or CheckerConfig()
    comp_text = flatten_messages(compressed)
    comp_norm = _normalize(comp_text)
    hits: list[Hit] = []
    preserved = fragmented = lost = 0

    for c in constraints:
        c_norm = _normalize(c.text)
        if c_norm in comp_norm:
            preserved += 1
            continue
        grams = _ngrams(c_norm)
        if grams:
            residual = {g for g in grams if g in comp_norm}
            overlap = len(residual) / len(grams)
        else:
            overlap = 1.0 if c_norm and c_norm in comp_norm else 0.0
        # 最长残留片段：从全长递减找第一个在压缩侧出现的子串
        fragment = ""
        for length in range(len(c_norm), 1, -1):
            found = False
            for i in range(len(c_norm) - length + 1):
                sub = c_norm[i:i + length]
                if sub in comp_norm:
                    fragment = sub
                    found = True
                    break
            if found:
                break
        msg_idx, off_s, off_e = _find_in_messages("", c.text, original)
        cm_idx, cm_off = _locate_in_compressed(comp_text, fragment or c_norm, compressed)
        if overlap >= cfg.constraint_overlap_threshold:
            fragmented += 1
            hits.append(Hit(
                checker="constraint", kind="constraint_fragmented",
                message_id=original[msg_idx].message_id if msg_idx is not None else None,
                message_index=msg_idx, offset_start=max(off_s, 0), offset_end=max(off_e, 0),
                snippet=c.text,
                detail=f'约束未完整保留，残留片段"{fragment}"，n-gram 重叠率 {overlap:.2f}',
                compressed_message_id=compressed[cm_idx].message_id if cm_idx is not None else None,
                compressed_message_index=cm_idx, compressed_offset_start=cm_off,
                extra={"overlap": round(overlap, 4), "fragment": fragment},
            ))
        else:
            lost += 1
            hits.append(Hit(
                checker="constraint", kind="constraint_lost",
                message_id=original[msg_idx].message_id if msg_idx is not None else None,
                message_index=msg_idx, offset_start=max(off_s, 0), offset_end=max(off_e, 0),
                snippet=c.text,
                detail=f'约束完全丢失，n-gram 重叠率 {overlap:.2f}',
                extra={"overlap": round(overlap, 4)},
            ))
    return CheckResult(
        checker="constraint", hit_count=len(hits), hits=hits,
        stats={"total_constraints": len(constraints),
               "preserved": preserved, "fragmented": fragmented, "lost": lost},
    )


# ---------------------------------------------------------------- 2. 否定词删除

def _predicate_head(text: str, pos: int, head_len: int) -> tuple[str, int] | None:
    """否定词之后 pos 处的谓词头：CJK 连续汉字（最多 head_len 字），英文取后接单词。

    返回 (head, delta)：delta 是 head 首字符相对 pos 的偏移（英文跳过的空格数）。
    """
    rest = text[pos:]
    if not rest:
        return None
    if is_cjk(rest[0]):
        head = ""
        for ch in rest:
            if is_cjk(ch) and len(head) < head_len:
                head += ch
            else:
                break
        return (head, 0) if head else None
    m = re.match(r"\s*([A-Za-z_]+)", rest)  # 先跳空格再取词
    return (m.group(1), m.start(1)) if m else None


def _extract_negation_spans(text: str, specs: list[NegationSpec],
                            stopphrases: set) -> list[tuple[str, str, int]]:
    """提取否定短语 (span, head, start)。停用搭配跳过；长词优先、区间消费防重复。"""
    blocked = [False] * len(text)
    for ph in sorted(stopphrases, key=len, reverse=True):
        for m in re.finditer(re.escape(ph), text):
            for i in range(m.start(), m.end()):
                blocked[i] = True
    spans = []
    for spec in sorted(specs, key=lambda s: -len(s.token)):
        for m in re.finditer(re.escape(spec.token), text):
            s, e = m.span()
            if any(blocked[s:e]):
                continue
            if is_cjk(spec.token[0]):
                # 前一字符非汉字/字母/数字（排除「差不多」里的「不」——停用表已挡，这里防粘连）
                prev = text[s - 1] if s > 0 else ""
                if prev and (is_cjk(prev) or prev.isalnum()):
                    continue
            else:
                if s > 0 and (text[s - 1].isalnum() or text[s - 1] == "_"):
                    continue
                if e < len(text) and (text[e].isalnum() or text[e] == "_"):
                    continue
            result = _predicate_head(text, e, spec.scope_head_len)
            if result is None:
                continue
            head, delta = result
            span_end = e + delta + len(head)
            for i in range(s, span_end):
                if i < len(blocked):
                    blocked[i] = True
            # span 从原文切片（英文否定词与谓词头之间可能有空格，拼接会丢）
            spans.append((text[s:span_end], head, s))
    return spans


def check_negations(original: list[MessageText], compressed: list[CompressedMessage],
                    specs: list[NegationSpec] | None = None,
                    config: CheckerConfig | None = None) -> CheckResult:
    cfg = config or CheckerConfig()
    specs = specs or cfg.negation_specs
    orig_text = "\n".join(m.text for m in original)
    comp_text = flatten_messages(compressed)

    orig_spans = _extract_negation_spans(orig_text, specs, cfg.negation_stopphrases)
    comp_spans = _extract_negation_spans(comp_text, specs, cfg.negation_stopphrases)

    hits: list[Hit] = []
    polarity_flips = negation_lost = 0

    for span, head, start in orig_spans:
        o_neg = orig_text.count(span)
        c_neg = comp_text.count(span)
        o_plain = orig_text.count(head) - o_neg          # 非谓词头位置的出现次数
        c_plain = comp_text.count(head) - c_neg
        flip = min(max(o_neg - c_neg, 0), max(c_plain - o_plain, 0))
        if flip > 0:
            polarity_flips += 1
            # 定位到原文侧消息
            msg_idx, off_s, off_e = _find_in_messages("", span, original)
            cm_idx, cm_off = _locate_in_compressed(comp_text, head, compressed)
            hits.append(Hit(
                checker="negation", kind="polarity_flip",
                message_id=original[msg_idx].message_id if msg_idx is not None else None,
                message_index=msg_idx,
                offset_start=off_s if off_s >= 0 else start,
                offset_end=(off_e if off_e >= 0 else start + len(span)),
                snippet=span,
                detail=f'否定短语"{span}"被压缩器剥掉否定词，谓词头"{head}"仍存活，语义极性翻转',
                compressed_message_id=compressed[cm_idx].message_id if cm_idx is not None else None,
                compressed_message_index=cm_idx, compressed_offset_start=cm_off,
                extra={"span": span, "head": head, "flip": flip,
                       "o_neg": o_neg, "c_neg": c_neg},
            ))
        elif o_neg - c_neg > 0:
            negation_lost += 1
            if cfg.report_lost_negations:
                msg_idx, off_s, off_e = _find_in_messages("", span, original)
                hits.append(Hit(
                    checker="negation", kind="negation_lost",
                    message_id=original[msg_idx].message_id if msg_idx is not None else None,
                    message_index=msg_idx,
                    offset_start=off_s if off_s >= 0 else start,
                    offset_end=(off_e if off_e >= 0 else start + len(span)),
                    snippet=span,
                    detail=f'否定短语"{span}"随所在内容整段丢失（谓词头同步消失，非极性翻转）',
                    extra={"span": span, "head": head},
                ))
    return CheckResult(
        checker="negation", hit_count=len(hits), hits=hits,
        stats={"negation_spans": len(orig_spans),
               "deleted_spans": negation_lost,
               "polarity_flips": polarity_flips,
               "negation_lost": negation_lost},
    )


# ---------------------------------------------------------------- 3. 在途工具链

def check_toolchain(original: list[MessageText],
                    compressed: list[CompressedMessage]) -> CheckResult:
    # 原始侧结构化提取（id 集合来自解析，绝不写死 id 正则）
    call_msg: dict = {}     # id -> (msg_index, msg_id, tool_name)
    result_msg: dict = {}   # id -> (msg_index, msg_id)
    for idx, m in enumerate(original):
        for tc in m.tool_calls:
            call_msg[tc.tool_use_id] = (idx, m.message_id, tc.name)
        for rid in m.tool_result_ids:
            result_msg[rid] = (idx, m.message_id)
    complete_ids = [i for i in call_msg if i in result_msg]

    # 压缩侧：结构化判定（优先）；全 None 时退化为文本出现次数
    cm_by_id = {cm.message_id: cm for cm in compressed if cm.message_id is not None}
    structured = bool(cm_by_id)

    def _call_survived(cid: str) -> bool:
        if structured:
            mid = call_msg[cid][1]
            cm = cm_by_id.get(mid)
            return cm is not None and cm.kind == "tool_call"
        return flatten_messages(compressed).count(cid) >= 1

    def _result_survived(cid: str) -> bool:
        if structured:
            mid = result_msg[cid][1]
            cm = cm_by_id.get(mid)
            return cm is not None and cm.kind == "tool_result"
        return flatten_messages(compressed).count(cid) >= 2

    hits: list[Hit] = []
    intact = removed = cwo = orphan = 0
    for cid in complete_ids:
        cs, rs = _call_survived(cid), _result_survived(cid)
        idx, mid, name = call_msg[cid]
        if cs and rs:
            intact += 1
        elif cs and not rs:
            cwo += 1
            hits.append(Hit(
                checker="toolchain", kind="call_without_result",
                message_id=mid, message_index=idx, offset_start=0, offset_end=0,
                snippet=cid,
                detail=f"工具调用 {name} 的调用在压缩后保留，但其 tool_result 被删："
                       f"在途工具链截断，agent 可能重放",
                compressed_message_index=0, compressed_offset_start=None,
                extra={"tool_name": name, "tool_use_id": cid},
            ))
        elif rs and not cs:
            orphan += 1
            hits.append(Hit(
                checker="toolchain", kind="orphan_result",
                message_id=result_msg[cid][1], message_index=result_msg[cid][0],
                offset_start=0, offset_end=0, snippet=cid,
                detail=f"工具调用 {name} 的调用被删，但 tool_result 仍留在上下文中（孤儿结果）",
                extra={"tool_name": name, "tool_use_id": cid},
            ))
        else:
            removed += 1
    return CheckResult(
        checker="toolchain", hit_count=len(hits), hits=hits,
        stats={"total_pairs": len(complete_ids), "intact_pairs": intact,
               "removed_pairs": removed, "call_without_result": cwo,
               "orphan_result": orphan},
    )


# ---------------------------------------------------------------- 4. 数值变形

def _mask_excluded(text: str, patterns: tuple) -> str:
    masked = text
    for pat in patterns:
        masked = re.sub(pat, lambda m: " " * (m.end() - m.start()), masked)
    return masked


def _extract_numbers(text: str, pattern: str, exclude: tuple) -> list[tuple[str, int, int]]:
    masked = _mask_excluded(text, exclude)
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, masked)]


def _number_value(raw: str) -> float:
    s = raw.replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100
    if s[-1] in "kKmM":
        mult = 1e3 if s[-1] in "kK" else 1e6
        return float(s[:-1]) * mult
    return float(s)


def check_numbers(original: list[MessageText], compressed: list[CompressedMessage],
                  config: CheckerConfig | None = None) -> CheckResult:
    cfg = config or CheckerConfig()
    comp_text = flatten_messages(compressed)

    # 原文数字池（按出现顺序）
    orig_nums: list[dict] = []
    for idx, m in enumerate(original):
        for raw, s, e in _extract_numbers(m.text, cfg.number_pattern, cfg.exclude_patterns):
            orig_nums.append({"raw": raw, "value": _number_value(raw),
                              "msg_index": idx, "msg_id": m.message_id,
                              "off_s": s, "off_e": e})
    # 压缩侧数字池（多重集 + 使用标记）
    pool = []
    pos = 0
    for cm_idx, cm in enumerate(compressed):
        for raw, s, e in _extract_numbers(cm.text, cfg.number_pattern, cfg.exclude_patterns):
            pool.append({"raw": raw, "value": _number_value(raw), "used": False,
                         "global_s": pos + s, "cm_idx": cm_idx, "off_s": s})
        pos += len(cm.text) + 1

    hits: list[Hit] = []
    preserved = approx = approx_word = lost = 0
    for n in orig_nums:
        # 1) 精确值相等
        match = None
        for p in pool:
            if not p["used"] and abs(p["value"] - n["value"]) < 1e-12:
                match = p
                break
        if match:
            match["used"] = True
            preserved += 1
            continue
        # 2) 近似：相对差最小且 ≤ band
        best = None
        for p in pool:
            if p["used"]:
                continue
            base = max(abs(n["value"]), 1e-12)
            rel = abs(p["value"] - n["value"]) / base
            if rel <= cfg.number_approx_band and (best is None or rel < best[0]):
                best = (rel, p)
        if best:
            rel, p = best
            p["used"] = True
            # 近似标记词：匹配数字前 3 字符内出现
            before = comp_text[max(0, p["global_s"] - 3):p["global_s"]]
            is_word = any(mk in before for mk in cfg.approx_markers)
            if is_word:
                approx_word += 1
                kind = "approx_word"
                detail = (f'精确数值 {n["raw"]} 被约化为 "{p["raw"]}"（带约化标记），'
                          f"相对差 {rel:.2%}")
            else:
                approx += 1
                kind = "approx"
                detail = (f'精确数值 {n["raw"]} 在压缩后变成近似值 {p["raw"]}'
                          f"（相对差 {rel:.2%}）")
            hits.append(Hit(
                checker="numeric", kind=kind,
                message_id=n["msg_id"], message_index=n["msg_index"],
                offset_start=n["off_s"], offset_end=n["off_e"], snippet=n["raw"],
                detail=detail,
                compressed_message_id=compressed[p["cm_idx"]].message_id,
                compressed_message_index=p["cm_idx"], compressed_offset_start=p["off_s"],
                extra={"original_value": n["value"], "matched": p["raw"],
                       "matched_value": p["value"], "rel_diff": round(rel, 6)},
            ))
        else:
            lost += 1
            hits.append(Hit(
                checker="numeric", kind="lost",
                message_id=n["msg_id"], message_index=n["msg_index"],
                offset_start=n["off_s"], offset_end=n["off_e"], snippet=n["raw"],
                detail=f'精确数值 {n["raw"]} 在压缩后完全消失',
                extra={"original_value": n["value"]},
            ))
    return CheckResult(
        checker="numeric", hit_count=len(hits), hits=hits,
        stats={"total_numbers": len(orig_nums), "preserved": preserved,
               "approx": approx, "approx_word": approx_word, "lost": lost},
    )


# ---------------------------------------------------------------- 聚合入口

def run_all_checks(original: list[MessageText], compressed: list[CompressedMessage],
                   constraints: list[Constraint] | None = None,
                   config: CheckerConfig | None = None) -> CheckReport:
    cfg = config or CheckerConfig()
    results = [
        check_constraints(original, compressed, constraints or [], cfg),
        check_negations(original, compressed, None, cfg),
        check_toolchain(original, compressed),
        check_numbers(original, compressed, cfg),
    ]
    return CheckReport(results=results, config=cfg.to_dict())


def run_all_checkers(before_msgs: list[Msg], after_msgs: list[Msg],
                     annotations: dict | None = None) -> dict:
    """模块 E 的契约适配：Msg 序列 + 任务注解 → 四项损坏计数。

    annotations = {"constraints": [str], ...}（negations/tool_pairs/numeric_facts
    由结构或全文本扫描推导，无需外部传入）。
    """
    annotations = annotations or {}
    original = msgs_to_message_texts(before_msgs)
    compressed = msgs_to_compressed(after_msgs)
    constraints = [Constraint(t) for t in annotations.get("constraints", [])]
    report = run_all_checks(original, compressed, constraints)
    return {r.checker: r.hit_count for r in report.results}


def json_loads_report(path: str) -> CheckReport:
    """从报告 JSON 反序列化（实验模块汇总用）。"""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    results = []
    for r in d["results"]:
        results.append(CheckResult(
            checker=r["checker"], hit_count=r["hit_count"],
            hits=[Hit(**h) for h in r["hits"]], stats=r["stats"],
        ))
    return CheckReport(results=results, config=d["config"])


def main(argv: list[str] | None = None) -> int:
    """CLI：ctxautopsy check --original O.json --compressed C.json
    --constraints X.json [--config cfg.json] [--out report.json]

    original/compressed JSON 形态均为 {"messages": [{"message_id","role","text",
    "tool_calls":[...],"tool_result_ids":[...]}]}（压缩侧可只含 message_id/text/kind）。
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="ctxautopsy check", description="四项损坏检查")
    ap.add_argument("--original", required=True)
    ap.add_argument("--compressed", required=True)
    ap.add_argument("--constraints", default=None, help="约束 JSON: [\"约束1\", ...]")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    def _load(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    od, cd = _load(args.original), _load(args.compressed)
    original = [MessageText(
        message_id=m.get("message_id", ""), role=m.get("role", ""), text=m.get("text", ""),
        tool_calls=[ToolCall(**tc) for tc in m.get("tool_calls", [])],
        tool_result_ids=m.get("tool_result_ids", []),
    ) for m in od["messages"]]
    compressed = [CompressedMessage(message_id=m.get("message_id"), text=m.get("text", ""),
                                    kind=m.get("kind")) for m in cd["messages"]]
    constraints = [Constraint(t) for t in (_load(args.constraints) if args.constraints else [])]
    report = run_all_checks(original, compressed, constraints)
    text = report.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"报告已写入 {args.out}，共 {report.total_hits} 处损坏")
    else:
        print(text)
    return 0
