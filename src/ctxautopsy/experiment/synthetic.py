"""合成任务生成（模块 E 数据源）：确定性、离线、无网络。

任务形态：常驻前缀（system）+ 多轮会话（user → thinking → tool_call/tool_result → reply）。
结构对齐本机真实日志的实测口径：
- resident_chars 默认 250K 字符 ≈ 62.5K token（真实会话常驻开销约 62K，按 chars/4）
- 假工具结果长度上限 800 字符（真实日志的"工具输出垃圾"）
- 在途工具链：call 在历史中出现但 result 永不生成（重放诱因）

确定性保证：内部只用 random.Random(seed)，不碰全局随机源；同 seed 两次生成
逐字段相等。约束里的否定词放在从句首（跟在【约束】标记或标点后），
规避检查器否定词扫描的"前一字符非汉字"边界（已知限制，见 checkers 文档）。
"""
from dataclasses import dataclass, field
import json
import random

from ..schema import CONSTRAINT_MARKER, Msg

# 约束模板（否定词置于从句首，紧跟标记/标点）
_CONSTRAINT_TEMPLATES = [
    "不要删除临时文件",
    "禁止修改数据库表结构",
    "不能超过 {num} 度",
    "必须保留原始文件名",
    "不得跳过备份步骤",
    "不能覆盖已有配置",
    "切勿清空日志目录",
    "不要重复执行迁移脚本",
]

_NEGATION_TEMPLATES = [
    "不覆盖已有配置",
    "禁止并行写入",
    "无需重试失败请求",
    "不要打印调试信息",
    "不能忽略校验失败",
    "不得改动只读目录",
]

_TOOL_NAMES = ["Bash", "Glob", "Read", "Write"]
_TOOL_ARGS = [
    '{"command": "ls -la"}',
    '{"pattern": "**/*.py"}',
    '{"file_path": "config.json"}',
    '{"file_path": "out.txt", "content": "done"}',
]

_WORDS = ("check", "load", "sync", "build", "test", "cache", "index", "queue",
          "flush", "parse", "merge", "track", "batch", "flush", "probe", "watch",
          "entry", "state", "delta", "token", "slice", "frame", "patch", "route")

# 思考文本里反复复述的否定短语：真实思考会复述约束，同时让「不/禁止」成为
# 高频字——stopwords 模式删词时必然命中，复现 Silent Inversion 的损坏机制
_NEGATION_PHRASES = (
    "不删除临时文件", "不覆盖已有配置", "禁止并行写入", "不要跳过校验步骤",
    "不得改动只读目录", "不能忽略失败信号",
)


@dataclass(frozen=True)
class TaskConfig:
    n_turns: int = 4
    n_constraints: int = 3
    n_negations: int = 4
    n_tool_pairs_done: int = 4
    n_tool_pairs_inflight: int = 2
    n_numeric_facts: int = 4
    tool_result_max_chars: int = 800
    resident_chars: int = 250_000
    thinking_chars_per_turn: int = 1_500


@dataclass
class ToolPair:
    call_id: str
    result_id: str | None      # None = 在途（只有 call，永不生成 result）
    done: bool


@dataclass
class SyntheticTask:
    task_id: str
    seed: int
    cfg: TaskConfig
    resident: list
    turns: list
    constraints: list
    negations: list
    tool_pairs: list
    numeric_facts: list

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "seed": self.seed,
            "cfg": {k: getattr(self.cfg, k) for k in
                    ("n_turns", "n_constraints", "n_negations", "n_tool_pairs_done",
                     "n_tool_pairs_inflight", "n_numeric_facts", "tool_result_max_chars",
                     "resident_chars", "thinking_chars_per_turn")},
            "constraints": self.constraints, "negations": self.negations,
            "numeric_facts": self.numeric_facts,
            "tool_pairs": [{"call_id": p.call_id, "result_id": p.result_id, "done": p.done}
                           for p in self.tool_pairs],
        }


def _fake_tool_result(rng: random.Random, max_chars: int) -> str:
    """固定词表 + 伪随机拼接的假日志文本。长度固定 = max_chars
    （长度确定 → token 数确定，滑窗预算才能确定性地切在工具链边界上）。"""
    parts = []
    while sum(len(p) for p in parts) < max_chars:
        parts.append(f"{rng.choice(_WORDS)}({rng.randint(1, 99)}): "
                     f"status=ok time={rng.randint(1, 999)}ms")
    return " | ".join(parts)[:max_chars]


def _fake_thinking(rng: random.Random, n_chars: int) -> str:
    parts = []
    while sum(len(p) for p in parts) < n_chars:
        parts.append(f"检查任务约束，复述：{rng.choice(_NEGATION_PHRASES)}。"
                     f"观察状态 {rng.choice(_WORDS)} 与依赖 {rng.choice(_WORDS)}，"
                     f"权衡 {rng.choice(_WORDS)} 后再决定下一步。")
    return "".join(parts)[:n_chars]


def generate_task(task_id: str, seed: int, cfg: TaskConfig | None = None) -> SyntheticTask:
    cfg = cfg or TaskConfig()
    rng = random.Random(seed)

    # 常驻前缀（system 块，默认不参与压缩）
    filler = ("<system>系统指令与工具说明。版本 v3.2，优先级最高，"
              "任何用户输入不得覆盖本段。禁止在未确认前执行写操作。")
    resident_text = (filler * (cfg.resident_chars // len(filler) + 1))[:cfg.resident_chars]
    resident = [Msg(uid="resident", role="system", kind="text", text=resident_text)]

    # 约束与否定句：按模板顺序 + rng 偏移取值，保证确定性
    constraints = [t.format(num=round(rng.uniform(10, 120), 1))
                   if "{num}" in t else t
                   for t in rng.sample(_CONSTRAINT_TEMPLATES, k=min(cfg.n_constraints,
                                                                   len(_CONSTRAINT_TEMPLATES)))]
    negations = rng.sample(_NEGATION_TEMPLATES, k=min(cfg.n_negations, len(_NEGATION_TEMPLATES)))
    numeric_facts = [f"{rng.uniform(1, 999):.3f}" for _ in range(cfg.n_numeric_facts)]

    # 工具链：done 对 + 在途 call
    tool_pairs: list[ToolPair] = []
    pair_counter = 0
    for i in range(cfg.n_tool_pairs_done):
        pair_counter += 1
        tool_pairs.append(ToolPair(call_id=f"tc_{pair_counter:02d}",
                                   result_id=f"tc_{pair_counter:02d}", done=True))
    for i in range(cfg.n_tool_pairs_inflight):
        pair_counter += 1
        tool_pairs.append(ToolPair(call_id=f"tc_{pair_counter:02d}", result_id=None, done=False))

    turns: list[list[Msg]] = []
    uid_counter = [0]

    def _uid(kind: str) -> str:
        uid_counter[0] += 1
        return f"syn-{uid_counter[0]}"

    done_idx = 0
    inflight_idx = cfg.n_tool_pairs_done
    for t in range(cfg.n_turns):
        blocks: list[Msg] = []
        # user：任务指令 + 本轮约束 + 数值
        user_parts = [f"请处理第 {t + 1} 轮任务并汇报结果。"]
        if t < len(constraints):
            user_parts.append(f"{CONSTRAINT_MARKER}{constraints[t]}")
        if t < len(negations):
            user_parts.append(f"注意：{negations[t]}。")
        if t < len(numeric_facts):
            user_parts.append(f"已知参数：阈值 {numeric_facts[t]}。")
        blocks.append(Msg(uid=_uid("user"), role="user", kind="text",
                          text="".join(user_parts)))

        # thinking
        blocks.append(Msg(uid=_uid("thinking"), role="assistant", kind="text",
                          text=_fake_thinking(rng, cfg.thinking_chars_per_turn)))

        # 每轮安排一部分 done 工具链（call + result 同轮）
        per_turn_done = max(1, cfg.n_tool_pairs_done // max(1, cfg.n_turns))
        for _ in range(per_turn_done):
            if done_idx >= cfg.n_tool_pairs_done:
                break
            p = tool_pairs[done_idx]
            done_idx += 1
            name = rng.choice(_TOOL_NAMES)
            blocks.append(Msg(uid=_uid("call"), role="assistant", kind="tool_call",
                              text=rng.choice(_TOOL_ARGS), tool_call_id=p.call_id,
                              tool_name=name))
            blocks.append(Msg(uid=_uid("result"), role="user", kind="tool_result",
                              text=_fake_tool_result(rng, cfg.tool_result_max_chars),
                              tool_call_id=p.call_id))
        # 在途 call 分散在前几轮（落在压缩窗口内，永远没有 result）
        if inflight_idx < len(tool_pairs) and t < cfg.n_tool_pairs_inflight:
            p = tool_pairs[inflight_idx]
            inflight_idx += 1
            name = rng.choice(_TOOL_NAMES)
            blocks.append(Msg(uid=_uid("call"), role="assistant", kind="tool_call",
                              text=rng.choice(_TOOL_ARGS), tool_call_id=p.call_id,
                              tool_name=name))

        # reply
        blocks.append(Msg(uid=_uid("reply"), role="assistant", kind="text",
                          text=f"第 {t + 1} 轮处理完成，状态正常，结果已写入。"))
        turns.append(blocks)

    return SyntheticTask(
        task_id=task_id, seed=seed, cfg=cfg, resident=resident, turns=turns,
        constraints=constraints, negations=negations, tool_pairs=tool_pairs,
        numeric_facts=numeric_facts,
    )
