"""按词频删词策略：模拟真实压缩器删高频/低频词的失败模式。

参数：mode（stopwords=删频次最高 | rare=删频次最低）、top_ratio（删词比例，相对词表大小）、
apply_to（作用于哪些 kind 的消息，默认 ["text"]，可加 "tool_result" 模拟
"压缩器连工具输出一起糟蹋"）。

分词：数字串/英文串为整词，每个 CJK 汉字为独立 token。
stopwords 模式下中文高频字（的/不/了）天然被删——否定词损坏由此复现；
rare 模式下孤立数字（如 3.14）被删——数值损坏由此复现。
压缩器不保护任何内容：约束文本与普通文本一视同仁，损坏与否由模块 D 判定。
"""
import math
import re

from ..schema import TOKEN_RE, ChangeEntry, Msg, VALID_KINDS
from .base import BaseCompressor


def tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_RE, text)


def drop_tokens_from_text(text: str, drop_set: set) -> str:
    """删除命中集合的 token，其余内容（标点、空白）原样保留。"""
    parts = []
    pos = 0
    for m in re.finditer(TOKEN_RE, text):
        parts.append(text[pos:m.start()])
        if m.group() not in drop_set:
            parts.append(m.group())
        pos = m.end()
    parts.append(text[pos:])
    return "".join(parts)


class FreqDropCompressor(BaseCompressor):
    name = "freq_drop"
    default_params = {"mode": "stopwords", "top_ratio": 0.05, "apply_to": ["text"]}

    def _compress(self, messages: list[Msg], params: dict) -> tuple[list[Msg], list]:
        mode = params["mode"]
        if mode not in ("stopwords", "rare"):
            raise ValueError(f"freq_drop 的 mode 必须是 stopwords/rare，收到: {mode}")
        apply_to = list(params["apply_to"])
        bad = [k for k in apply_to if k not in VALID_KINDS]
        if bad:
            raise ValueError(f"freq_drop 的 apply_to 含非法 kind: {bad}，合法值: {list(VALID_KINDS)}")

        # 词频统计（语料 = apply_to 指定的消息文本）
        counts: dict = {}
        for m in messages:
            if m.kind in apply_to:
                for tok in tokenize(m.text):
                    counts[tok] = counts.get(tok, 0) + 1

        if not counts:
            return list(messages), []
        # 同频平手按 (-count, token) 排序保证确定性；rare 模式取 (count, token) 升序
        # （低频组里 token 字典序最小者先删——孤立数字 "3.14" 排在 CJK 字之前）
        drop_count = max(1, math.ceil(len(counts) * float(params["top_ratio"])))
        if mode == "stopwords":
            vocab = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            drop_set = {tok for tok, _ in vocab[:drop_count]}
            reason = "freq_drop_stopwords"
        else:
            vocab = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
            drop_set = {tok for tok, _ in vocab[:drop_count]}
            reason = "freq_drop_rare"

        out: list[Msg] = []
        changes: list[ChangeEntry] = []
        for m in messages:
            if m.kind not in apply_to:
                out.append(m)
                continue
            new_text = drop_tokens_from_text(m.text, drop_set)
            if new_text == m.text:
                out.append(m)
                continue
            # 删空剩标点/空白 → 升级为 delete
            if not tokenize(new_text):
                changes.append(ChangeEntry(
                    uid=m.uid, op="delete", kind=m.kind,
                    orig_text=m.text, new_text=None, reason=reason,
                    meta={"dropped_tokens": tokenize(m.text)},
                ))
                continue
            dropped = [t for t in tokenize(m.text) if t in drop_set]
            out.append(Msg(uid=m.uid, role=m.role, kind=m.kind, text=new_text,
                           tool_call_id=m.tool_call_id, tool_name=m.tool_name,
                           tokens=m.tokens, meta=m.meta))
            changes.append(ChangeEntry(
                uid=m.uid, op="modify", kind=m.kind,
                orig_text=m.text, new_text=new_text, reason=reason,
                meta={"dropped_tokens": dropped},
            ))
        return out, changes
