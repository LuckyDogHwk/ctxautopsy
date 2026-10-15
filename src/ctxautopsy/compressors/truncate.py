"""截断策略：保头删尾 / 保尾删头。

参数：mode（head|tail，注册名预设）、keep_items（保留条数）、keep_ratio（保留比例，
keep_items 优先；保留条数 = keep_items 若给出，否则 ceil(n × keep_ratio)）。
被删条目 op="delete"、orig_text 全量保留。保留数超过序列长度时原样返回、无变更。
keep_items=0 是合法值（100% 压缩，实验需要它当上界对照）。
"""
import math

from ..schema import ChangeEntry, Msg
from .base import BaseCompressor


class TruncateCompressor(BaseCompressor):
    name = "truncate"
    default_params = {"mode": "head", "keep_items": None, "keep_ratio": 0.5}

    def _compress(self, messages: list[Msg], params: dict) -> tuple[list[Msg], list]:
        mode = params["mode"]
        n = len(messages)
        if params["keep_items"] is not None:
            keep = int(params["keep_items"])
        else:
            keep = math.ceil(n * float(params["keep_ratio"]))

        if keep >= n:
            return list(messages), []

        kept = messages[:keep] if mode == "head" else messages[n - keep:]
        kept_uids = {m.uid for m in kept}
        reason = f"truncate_{mode}"
        changes = [
            ChangeEntry(uid=m.uid, op="delete", kind=m.kind,
                        orig_text=m.text, new_text=None, reason=reason)
            for m in messages if m.uid not in kept_uids
        ]
        return kept, changes
