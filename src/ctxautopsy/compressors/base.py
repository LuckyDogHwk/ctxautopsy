"""压缩器基类：校验、参数合并、统计与不变式，"作案"逻辑由子类实现。

子类只需实现 _compress(messages, params) -> (输出消息, 变更记录)，
以下四件事由基类免费提供：
1. 输入校验：uid 非空且全序列唯一、kind 合法、tool_call 必有 tool_call_id、
   未知参数名报错——任何违规抛 ValueError 并列出问题，绝不静默修复
2. 参数合并：default_params 为底、调用参数覆盖
3. 不变式断言：输出 uid 是输入 uid 的保序子序列；modify 条目的 new_text 与
   输出同 uid 消息的 text 逐字一致；每条被删/被改消息恰好一条 ChangeEntry
4. CompressStats 组装（纯数量字段，不写任何损坏结论）

全部策略为纯函数：同输入同参数两次调用结果逐字一致（无随机、无 LLM、无时间依赖）。
"""
from abc import ABC, abstractmethod
import math

from ..schema import VALID_KINDS, CompressResult, CompressStats, Msg, msg_tokens


class BaseCompressor(ABC):
    name: str = ""
    default_params: dict = {}

    def compress(self, messages: list[Msg], params: dict | None = None) -> CompressResult:
        merged = dict(self.default_params)
        if params:
            unknown = set(params) - set(self.default_params)
            if unknown:
                raise ValueError(
                    f"策略 {self.name} 的未知参数: {sorted(unknown)}，"
                    f"合法参数: {sorted(self.default_params)}"
                )
            merged.update(params)

        # ---- 输入校验 ----
        seen: set = set()
        problems = []
        for m in messages:
            if not m.uid:
                problems.append(f"消息 uid 为空: {m}")
            elif m.uid in seen:
                problems.append(f"消息 uid 重复: {m.uid}")
            seen.add(m.uid)
            if m.kind not in VALID_KINDS:
                problems.append(f"消息 {m.uid} 的 kind 非法: {m.kind}")
            if m.kind == "tool_call" and not m.tool_call_id:
                problems.append(f"tool_call 消息 {m.uid} 缺少 tool_call_id")
        if problems:
            raise ValueError("输入校验失败: " + "; ".join(problems))

        out_msgs, changes = self._compress(messages, merged)

        # ---- 不变式断言 ----
        in_uids = [m.uid for m in messages]
        out_uids = [m.uid for m in out_msgs]
        # 保序子序列
        it = iter(in_uids)
        if not all(uid in it for uid in out_uids):
            raise AssertionError("不变式违反: 输出 uid 不是输入 uid 的保序子序列")
        # modify 的 new_text 与输出逐字一致
        by_uid = {m.uid: m for m in out_msgs}
        for c in changes:
            if c.op == "modify":
                target = by_uid.get(c.uid)
                if target is None or target.text != c.new_text:
                    raise AssertionError(f"不变式违反: modify 条目 {c.uid} 的 new_text 与输出不一致")
            elif c.op == "delete":
                if c.uid in by_uid:
                    raise AssertionError(f"不变式违反: delete 条目 {c.uid} 的消息仍在输出中")
            else:
                raise AssertionError(f"不变式违反: 未知 op {c.op}")
        # 一一对应：每条被删/被改消息恰好一条 ChangeEntry
        changed_uids = {c.uid for c in changes}
        for uid in set(in_uids) - set(out_uids):
            if uid not in changed_uids:
                raise AssertionError(f"不变式违反: 消息 {uid} 被删除但无 ChangeEntry")
        for c in changes:
            if c.op == "modify" and c.uid not in set(in_uids).intersection(out_uids):
                raise AssertionError(f"不变式违反: modify 条目 {c.uid} 对应消息不在输入输出中")

        stats = self._build_stats(messages, out_msgs, changes)
        return CompressResult(
            strategy=self.name, params=merged,
            messages=out_msgs, changes=changes, stats=stats,
        )

    @abstractmethod
    def _compress(self, messages: list[Msg], params: dict) -> tuple[list[Msg], list]:
        """子类实现：返回 (压缩后消息, 变更记录列表)。"""

    @staticmethod
    def _build_stats(in_msgs, out_msgs, changes) -> CompressStats:
        input_tokens = sum(msg_tokens(m) for m in in_msgs)
        output_tokens = sum(msg_tokens(m) for m in out_msgs)
        deleted = sum(1 for c in changes if c.op == "delete")
        modified = sum(1 for c in changes if c.op == "modify")
        saved = max(0, input_tokens - output_tokens)
        return CompressStats(
            input_items=len(in_msgs), output_items=len(out_msgs),
            deleted_items=deleted, modified_items=modified,
            input_tokens=input_tokens, output_tokens=output_tokens,
            saved_tokens=saved,
            saved_ratio=(saved / input_tokens) if input_tokens > 0 else 0.0,
        )
