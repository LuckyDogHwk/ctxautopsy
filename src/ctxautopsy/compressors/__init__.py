"""压缩器注册表（模块 C）。

五个注册策略（全确定性、纯函数）：
- truncate_head / truncate_tail：保头删尾 / 保尾删头
- sliding_window：尾部 token 预算滑窗
- freq_drop：按词频删词（stopwords=删高频 / rare=删低频）
- rule_summary：长消息压成首句摘要，可选数字约化

分工声明：压缩器是"作案者"，不保护任何内容（约束、否定词、工具链、数值
一律与普通文本同等对待），损坏与否由模块 D 的检查器判定。
"""
from .base import BaseCompressor
from .freq_drop import FreqDropCompressor
from .rule_summary import RuleSummaryCompressor
from .sliding_window import SlidingWindowCompressor
from .truncate import TruncateCompressor


def _make_truncate(mode: str):
    def factory():
        c = TruncateCompressor()
        c.default_params = {"mode": mode, "keep_items": None, "keep_ratio": 0.5}
        c.name = f"truncate_{mode}"
        return c
    return factory


COMPRESSOR_REGISTRY: dict = {
    "truncate_head": _make_truncate("head"),
    "truncate_tail": _make_truncate("tail"),
    "sliding_window": SlidingWindowCompressor,
    "freq_drop": FreqDropCompressor,
    "rule_summary": RuleSummaryCompressor,
}


def create_compressor(name: str, params: dict | None = None) -> BaseCompressor:
    """按注册名创建压缩器实例；params 在此处预设（compress 调用处的参数仍可覆盖）。未知名抛 KeyError。"""
    if name not in COMPRESSOR_REGISTRY:
        raise KeyError(f"未知压缩策略: {name}，可选: {list_compressors()}")
    entry = COMPRESSOR_REGISTRY[name]
    c = entry() if callable(entry) else entry()
    if params:
        c.default_params = {**c.default_params, **params}
    return c


def list_compressors() -> list[str]:
    return sorted(COMPRESSOR_REGISTRY)


__all__ = [
    "BaseCompressor", "COMPRESSOR_REGISTRY", "create_compressor", "list_compressors",
    "TruncateCompressor", "SlidingWindowCompressor", "FreqDropCompressor",
    "RuleSummaryCompressor",
]
