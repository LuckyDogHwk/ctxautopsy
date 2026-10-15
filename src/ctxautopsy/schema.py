"""共享数据模型与常量（模块 A/C/D/E 共用）。

- Msg：会话消息单元，模块 C 压缩器的输入输出，模块 E 合成任务的载体
- ChangeEntry / CompressStats / CompressResult：压缩产物与变更账（"账本"）
- estimate_tokens：切点工具（ceil(len/4)），只用于相对比较，不是计费依据——
  真实四档计费在模块 B 用 jsonl 里的 usage 实测数据，两者解耦

分工原则：C 是"作案者"（压缩器不保护任何内容），D 是"法医"（损坏判定），
CompressStats 只含数量字段、ChangeEntry 只描述动作，不写任何损坏结论。
"""
from dataclasses import dataclass, field

# 约束注入标记（模块 E 注入、模块 D 定位时使用同一常量）
CONSTRAINT_MARKER = "【约束】"

# 数字正则（模块 D 数值检查用）：排除日期形态由 checkers 的 exclude_patterns 处理
DEFAULT_NUMBER_PATTERN = r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:%|[kKmM])?(?![\w])"

# 分词正则（模块 C 按词频删词用）：数字串/英文串为整词，每个 CJK 汉字为独立 token
TOKEN_RE = r"[0-9]+(?:\.[0-9]+)?|[A-Za-z_]+|[一-鿿]"

VALID_KINDS = ("text", "tool_call", "tool_result")


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


@dataclass(frozen=True)
class Msg:
    """会话消息单元。

    uid：全序列唯一 ID（真实日志用 jsonl 行 uuid 派生，合成数据用 "syn-N"），
         是 ChangeEntry 回指原消息的唯一键，压缩器绝不改写。
    kind：text / tool_call / tool_result。tool_call 的 text 是入参 JSON 字符串；
          tool_result 的 tool_call_id 是配对的调用 id（配对靠字段不靠解析文本）。
    tokens：模块 A 可填真实值；None 时压缩器用 estimate_tokens 兜底。
    """
    uid: str
    role: str                        # system / user / assistant
    kind: str                        # text / tool_call / tool_result
    text: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tokens: int | None = None
    meta: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """切点工具：ceil(len/4)，空串为 0。

    注意与 session_log.estimate_tokens（CJK 感知）并存：模块 A 的构成桶用后者
    （按本机真实日志校准），压缩器/实验的切点与压缩率用本函数（与缓存模型
    chars_per_token=4 口径一致）。两个函数各自标注用途，不混用。
    """
    if not text:
        return 0
    return max(1, -(-len(text) // 4))


def msg_tokens(msg: Msg) -> int:
    """消息 token 数：优先用字段里的真实值，缺失时估算。"""
    return msg.tokens if msg.tokens is not None else estimate_tokens(msg.text)


@dataclass
class ChangeEntry:
    """一条变更记录：被删/被改消息的全量信息。orig_text 删除时也全量保留
    （模块 D 的检查器需要原文做文本级比对，宁可冗余不可缺失）。"""
    uid: str
    op: str                        # delete | modify
    kind: str
    orig_text: str
    new_text: str | None           # modify 时的新文本；delete 时为 None
    reason: str                    # 机器可读原因（truncate_head / sliding_window / freq_drop_stopwords / …）
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "op": self.op, "kind": self.kind,
            "orig_text": self.orig_text, "new_text": self.new_text,
            "reason": self.reason, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChangeEntry":
        return cls(uid=d["uid"], op=d["op"], kind=d["kind"], orig_text=d["orig_text"],
                   new_text=d["new_text"], reason=d["reason"], meta=d.get("meta", {}))


@dataclass
class CompressStats:
    """纯数量统计。刻意不含任何"损坏"字段——判定是模块 D 的职责。"""
    input_items: int = 0
    output_items: int = 0
    deleted_items: int = 0
    modified_items: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    saved_tokens: int = 0
    saved_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "input_items": self.input_items, "output_items": self.output_items,
            "deleted_items": self.deleted_items, "modified_items": self.modified_items,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "saved_tokens": self.saved_tokens, "saved_ratio": self.saved_ratio,
        }


@dataclass
class CompressResult:
    """压缩器的唯一返回值。"""
    strategy: str
    params: dict
    messages: list
    changes: list
    stats: CompressStats

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "params": self.params,
            "messages": [{
                "uid": m.uid, "role": m.role, "kind": m.kind, "text": m.text,
                "tool_call_id": m.tool_call_id, "tool_name": m.tool_name, "tokens": m.tokens,
            } for m in self.messages],
            "changes": [c.to_dict() for c in self.changes],
            "stats": self.stats.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "CompressResult":
        return cls(
            strategy=d["strategy"],
            params=d["params"],
            messages=[Msg(uid=m["uid"], role=m["role"], kind=m["kind"], text=m["text"],
                          tool_call_id=m["tool_call_id"], tool_name=m["tool_name"],
                          tokens=m["tokens"]) for m in d["messages"]],
            changes=[ChangeEntry.from_dict(c) for c in d["changes"]],
            stats=CompressStats(**d["stats"]),
        )
