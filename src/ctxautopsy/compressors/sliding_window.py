"""滑窗策略：只保留尾部 token 预算内的内容。

参数：max_tokens（预算）、cut_boundary（预算边界恰好落在某条消息上时是否掐头保留，
True 时该消息 op="modify"、new_text 为原文尾段；掐头后为空则升级为 delete）。
掐头可能落在 tool_call 上把 args JSON 截坏——这是刻意保留的失败模式
（模块 D 的 json.loads 校验会抓到），uid/tool_call_id 字段永不受损。
"""
from ..schema import ChangeEntry, Msg, estimate_tokens, msg_tokens
from .base import BaseCompressor


class SlidingWindowCompressor(BaseCompressor):
    name = "sliding_window"
    default_params = {"max_tokens": 2000, "cut_boundary": True}

    def _compress(self, messages: list[Msg], params: dict) -> tuple[list[Msg], list]:
        budget = int(params["max_tokens"])
        cut_boundary = bool(params["cut_boundary"])
        out: list[Msg] = []
        changes: list[ChangeEntry] = []
        used = 0

        for m in reversed(messages):
            t = msg_tokens(m)
            if used + t <= budget:
                out.append(m)
                used += t
            elif cut_boundary and used < budget:
                remaining = budget - used
                # 保留末尾 remaining×4 个字符（estimate_tokens 为 ceil(len/4)，
                # len ≤ remaining×4 保证 token 数 ≤ remaining）
                tail = m.text[-remaining * 4:]
                if tail and tail != m.text:
                    out.append(Msg(uid=m.uid, role=m.role, kind=m.kind, text=tail,
                                   tool_call_id=m.tool_call_id, tool_name=m.tool_name,
                                   tokens=estimate_tokens(tail), meta=m.meta))
                    changes.append(ChangeEntry(
                        uid=m.uid, op="modify", kind=m.kind,
                        orig_text=m.text, new_text=tail,
                        reason="window_trim",
                    ))
                    used = budget
                else:
                    changes.append(ChangeEntry(
                        uid=m.uid, op="delete", kind=m.kind,
                        orig_text=m.text, new_text=None,
                        reason="sliding_window",
                    ))
            else:
                changes.append(ChangeEntry(
                    uid=m.uid, op="delete", kind=m.kind,
                    orig_text=m.text, new_text=None,
                    reason="sliding_window",
                ))

        out.reverse()
        # changes 顺序与输入顺序一致（被删部分在前，掐头消息是窗口最旧一条）
        changes.reverse()
        return out, changes
