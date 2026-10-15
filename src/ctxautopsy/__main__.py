"""CtxAutopsy（上下文验尸官）命令行入口。

用法：
  python -m ctxautopsy analyze [目录]            # 解剖真实会话日志 → 聚合统计 + 成本报告
  python -m ctxautopsy experiment [--config X]   # 合成任务 × 压缩策略对照实验
  python -m ctxautopsy check --original O.json --compressed C.json   # 四项损坏检查
"""
import argparse
import sys
from pathlib import Path


def _cmd_analyze(args) -> int:
    from .real_report import render_real_report
    from .session_log import parse_sessions_dir, sessions_to_json
    from .billing import cost_sessions, costs_to_json, DEFAULT_PRICES

    root = Path(args.dir)
    reports = parse_sessions_dir(root)
    if not reports:
        print(f"{root} 下没有找到会话 jsonl")
        return 1
    md = render_real_report(reports)

    if args.out:
        out = Path(args.out)
        out.write_text(md, encoding="utf-8")
        if out.suffix == ".json":
            out.write_text(sessions_to_json(reports) + "\n" + costs_to_json(
                cost_sessions(reports, DEFAULT_PRICES)), encoding="utf-8")
        print(f"报告已写入 {args.out}")
    else:
        print(md)
    return 0


def _cmd_check(args) -> int:
    from .checkers import main as check_main
    return check_main(args.check_args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ctxautopsy", description="上下文验尸官")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("analyze", help="解剖真实会话日志（只输出聚合数字）")
    p1.add_argument("dir", nargs="?", default=str(Path.home() / ".claude" / "projects"))
    p1.add_argument("--out", default=None, help="输出 JSON 报告路径")
    p1.set_defaults(func=_cmd_analyze)

    p2 = sub.add_parser("experiment", help="合成任务对照实验")
    p2.add_argument("--config", default=None)
    p2.add_argument("--out-dir", default=None)
    p2.set_defaults(func=lambda a: _cmd_experiment(a))

    p3 = sub.add_parser("check", help="四项损坏检查（ctxautopsy check --help 看参数）")
    p3.add_argument("check_args", nargs=argparse.REMAINDER)
    p3.set_defaults(func=_cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


def _cmd_experiment(args) -> int:
    from .experiment.report import main as exp_main
    import sys as _sys
    extra = []
    if args.config:
        extra += ["--config", args.config]
    if args.out_dir:
        extra += ["--out-dir", args.out_dir]
    return exp_main(extra or None)


if __name__ == "__main__":
    sys.exit(main())
