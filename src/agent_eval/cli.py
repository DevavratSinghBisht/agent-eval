"""agent-eval command line.

  agent-eval benchmarks | toolsets                 list registered benchmarks / toolsets and tools
  agent-eval prepare <benchmark_id | experiment.yaml>
  agent-eval run <experiment.yaml> [--retry-errored] [--no-finalize]
  agent-eval finalize <campaign_dir>
  agent-eval validate <campaign_dir>
  agent-eval report <campaign_dir>
  agent-eval label <campaign_dir> [--model claude-haiku-5-5] [--limit N]
  agent-eval trajectory <campaign_dir> <run_id> <task_id> [--html out.html]
  agent-eval compare <campaign_dir> <config_hash_a> <config_hash_b>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="agent-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("benchmarks"); sub.add_parser("toolsets")
    p = sub.add_parser("prepare"); p.add_argument("target", help="benchmark id, or an experiment YAML (for plugins with params)")
    p.add_argument("--data-dir", default="data")
    p = sub.add_parser("run"); p.add_argument("experiment"); p.add_argument("--retry-errored", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    for name in ("finalize", "validate", "report"):
        p = sub.add_parser(name); p.add_argument("campaign_dir")
    p = sub.add_parser("label"); p.add_argument("campaign_dir"); p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--provider", default="anthropic"); p.add_argument("--limit", type=int)
    p = sub.add_parser("trajectory"); p.add_argument("campaign_dir"); p.add_argument("run_id"); p.add_argument("task_id")
    p.add_argument("--html")
    p = sub.add_parser("compare"); p.add_argument("campaign_dir"); p.add_argument("a"); p.add_argument("b")
    args = ap.parse_args(argv)

    if args.cmd == "benchmarks":
        from .benchmarks import available

        print("\n".join(available()))
        print("(or any 'package.module:ClassName' subclass of agent_eval.benchmarks.Benchmark)")
        return 0
    if args.cmd == "toolsets":
        from .tools import DEFAULT_TOOLSETS, available_tools

        for name, tools in DEFAULT_TOOLSETS.items():
            print(f"{name:16s} {', '.join(tools) or '(no tools)'}")
        print("\nregistered tools:", ", ".join(available_tools()))
        return 0
    if args.cmd == "prepare":
        from .benchmarks import get_benchmark
        from .core.config import BenchmarkConfig, load_experiment

        if args.target.endswith((".yaml", ".yml")):
            cfg = load_experiment(args.target).benchmark
        else:
            cfg = BenchmarkConfig(id=args.target, data_dir=args.data_dir)
        bench = get_benchmark(cfg)
        bench.prepare()
        print(f"prepared {bench.id}: {len(bench.load_tasks())} tasks")
        return 0

    if args.cmd == "run":
        from .core.config import load_experiment
        from .runner.campaign import Campaign

        exp = load_experiment(args.experiment)
        camp = Campaign(exp)
        if args.retry_errored:
            for f in camp.work.glob("*/tasks/*.json"):
                if json.loads(f.read_text(encoding="utf-8"))["task"]["task_status"] in ("errored", "grading_error"):
                    f.unlink()
        asyncio.run(camp.run())
        if not args.no_finalize:
            return _finalize_validate_report(camp.dir)
        return 0

    if args.cmd == "finalize":
        return _finalize_validate_report(Path(args.campaign_dir))
    if args.cmd == "validate":
        from .analysis.validate import validate

        res = validate(args.campaign_dir)
        _print_validation(res)
        return 0 if res["ok"] else 1
    if args.cmd == "report":
        from .analysis.report import build_report

        print(build_report(args.campaign_dir))
        return 0
    if args.cmd == "label":
        from .analysis.labeler import label_campaign
        from .analysis.report import build_report

        out = asyncio.run(label_campaign(args.campaign_dir, args.model, args.provider, args.limit))
        build_report(args.campaign_dir)
        print(out)
        return 0
    if args.cmd == "trajectory":
        from .tracing.store import get_trajectory

        traj = get_trajectory(args.campaign_dir, args.run_id, args.task_id)
        if traj is None:
            print(f"no trajectory for {args.run_id} / {args.task_id}", file=sys.stderr)
            return 1
        if args.html:
            from .analysis.viewer import render

            print(render(traj, Path(args.html)))
        else:
            print(json.dumps(traj, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.cmd == "compare":
        import pandas as pd

        from .analysis.metrics import paired_compare

        d = Path(args.campaign_dir)
        print(json.dumps(paired_compare(pd.read_csv(d / "tasks.csv"), pd.read_csv(d / "runs.csv"), args.a, args.b), indent=1))
        return 0
    return 2


def _finalize_validate_report(d: Path) -> int:
    from .analysis.report import build_report
    from .analysis.validate import validate
    from .tracing.store import finalize

    finalize(d)
    res = validate(d)
    _print_validation(res)
    print("report:", build_report(d))
    return 0 if res["ok"] else 1


def _print_validation(res: dict) -> None:
    print(f"validation: {'OK' if res['ok'] else 'FAILED'} ({res['n_errors']} errors, {res['n_warnings']} warnings)")
    for f in res["findings"][:20]:
        print(f"  [{f['severity']}] {f['check']} {f['run_id'] or ''} {f['task_id'] or ''}: {f['detail']}")


if __name__ == "__main__":
    sys.exit(main())
