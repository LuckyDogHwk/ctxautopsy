"""规则摘要策略：把长消息压成「首句 + 原长度标注」，可选先对数字做约化。

参数：min_chars（只处理超过该长度的消息）、summary_len（保留首部字符数）、
approx_numbers（先对原文所有数字做 2 位有效数字约化再摘要，模拟"精确数值被约化"）、
apply_to（作用于哪些 kind，默认 ["text"]）。

句边界取 。！？!?.\n 中最早出现者；无句边界（中文长句）直接按字符截断（CJK 安全）。
约化用 decimal.ROUND_HALF_UP 写死——内置 round 是银行家舍入，会破坏确定性。
压缩器不保护约束：约束文本与普通文本一视同仁。
"""
import re
from decimal import ROUND_HALF_UP, Decimal

from ..schema import ChangeEntry, Msg, VALID_KINDS
from .base import BaseCompressor

_PLAIN_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def approx_number(num_str: str) -> str:
    """2 位有效数字、ROUND_HALF_UP：1234.567→约1200、0.05→约0.05。"""
    d = Decimal(num_str)
    if d == 0:
        return "约0"
    exp = d.adjusted()  # 首位数字的 10 次幂位置
    quantum = Decimal(1).scaleb(exp - 1)  # 10^(exp-1)：保留 2 位有效数字
    r = d.quantize(quantum, rounding=ROUND_HALF_UP)
    s = format(r, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"约{s}"


class RuleSummaryCompressor(BaseCompressor):
    name = "rule_summary"
    default_params = {"min_chars": 200, "summary_len": 80,
                      "approx_numbers": False, "apply_to": ["text"]}

    def _compress(self, messages: list[Msg], params: dict) -> tuple[list[Msg], list]:
        apply_to = set(params["apply_to"])
        bad = [k for k in apply_to if k not in VALID_KINDS]
        if bad:
            raise ValueError(f"rule_summary 的 apply_to 含非法 kind: {bad}")
        min_chars = int(params["min_chars"])
        summary_len = int(params["summary_len"])
        approx = bool(params["approx_numbers"])

        out: list[Msg] = []
        changes: list[ChangeEntry] = []
        for m in messages:
            if m.kind not in apply_to or len(m.text) <= min_chars:
                out.append(m)
                continue

            text = m.text
            if approx:
                text = _PLAIN_NUM_RE.sub(lambda mt: approx_number(mt.group()), text)

            # 句边界：。！？!?\n 中最早出现者；"." 仅当不是小数点（前后不同时为数字）时算边界
            boundaries = [text.find(ch) for ch in "。！？!?\n" if text.find(ch) >= 0]
            dot = text.find(".")
            while dot >= 0 and dot > 0 and dot + 1 < len(text) \
                    and text[dot - 1].isdigit() and text[dot + 1].isdigit():
                dot = text.find(".", dot + 1)
            if dot >= 0:
                boundaries.append(dot)
            cut = min(boundaries) + 1 if boundaries else -1
            if cut <= 0:
                head = text[:summary_len]
            else:
                head = text[:cut]
                if len(head) > summary_len:
                    head = head[:summary_len]
            new_text = head + f"……（原{len(m.text)}字，已摘要）"

            out.append(Msg(uid=m.uid, role=m.role, kind=m.kind, text=new_text,
                           tool_call_id=m.tool_call_id, tool_name=m.tool_name,
                           tokens=m.tokens, meta=m.meta))
            changes.append(ChangeEntry(
                uid=m.uid, op="modify", kind=m.kind,
                orig_text=m.text, new_text=new_text,
                reason="rule_summary",
                meta={"approx_numbers": approx},
            ))
        return out, changes
