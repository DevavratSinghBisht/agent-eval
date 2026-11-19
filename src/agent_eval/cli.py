"""agent-eval command line.

  agent-eval prepare browsecomp|browsecomp_plus
  agent-eval run <experiment.yaml> [--retry-errored] [--no-finalize]
  agent-eval finalize <campaign_dir>
  agent-eval validate <campaign_dir>
  agent-eval report <campaign_dir>
  agent-eval label <campaign_dir> [--model claude-haiku-5-5] [--limit N]
  agent-eval trajectory <campaign_dir> <run_id> <task_id> [--html out.html]
  agent-eval compare <campaign_dir> <config_hash_a> <config_hash_b>
  agent-eval rejudge <campaign_dir> --provider P --model M [--effort E|none] [--base-url U] [--name N]
  agent-eval agreement <campaign_dir> [--human labels.csv]
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
    p = sub.add_parser("prepare"); p.add_argument("benchmark"); p.add_argument("--data-dir", default="data")
    p = sub.add_parser("run"); p.add_argument("experiment"); p.add_argument("--retry-errored", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    for name in ("finalize", "validate", "report"):
        p = sub.add_parser(name); p.add_argument("campaign_dir")
    p = sub.add_parser("label"); p.add_argument("campaign_dir"); p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--provider", default="anthropic"); p.add_argument("--limit", type=int)
    p.add_argument("--base-url"); p.add_argument("--effort", default="low", help="'none' for models without reasoning controls")
    p = sub.add_parser("trajectory"); p.add_argument("campaign_dir"); p.add_argument("run_id"); p.add_argument("task_id")
    p.add_argument("--html")
    p = sub.add_parser("compare"); p.add_argument("campaign_dir"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("rejudge", help="grade stored responses with an extra judge (no agent calls)")
    p.add_argument("campaign_dir"); p.add_argument("--provider", required=True); p.add_argument("--model", required=True)
    p.add_argument("--effort", default="low", help="'none' for models without reasoning controls")
    p.add_argument("--base-url"); p.add_argument("--api-key-env"); p.add_argument("--name")
    p.add_argument("--concurrency", type=int, default=8); p.add_argument("--limit", type=int)
    p = sub.add_parser("agreement", help="inter-judge agreement (Cohen's kappa, Gwet's AC1, ...)")
    p.add_argument("campaign_dir"); p.add_argument("--human", help="CSV of hand labels: run_id, task_id, verdict")
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        if args.benchmark == "browsecomp":
            from .benchmarks.browsecomp import download

            print(download(args.data_dir))
        elif args.benchmark == "browsecomp_plus":
            from .benchmarks.browsecomp_plus import prepare

            prepare(args.data_dir)
            print("prepared", Path(args.data_dir) / "browsecomp_plus")
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

        effort = None if args.effort == "none" else args.effort
        out = asyncio.run(label_campaign(args.campaign_dir, args.model, args.provider, args.base_url, effort, args.limit))
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
    if args.cmd == "rejudge":
        from .core.config import GraderConfig
        from .runner.rejudge import rejudge

        cfg = GraderConfig(provider=args.provider, model=args.model, effort=None if args.effort == "none" else args.effort,
                           base_url=args.base_url, api_key_env=args.api_key_env, name=args.name)
        asyncio.run(rejudge(args.campaign_dir, cfg, args.concurrency, args.limit))
        return _finalize_validate_report(Path(args.campaign_dir))
    if args.cmd == "agreement":
        from .analysis.agreement import write_agreement
        from .analysis.report import build_report

        res = write_agreement(args.campaign_dir, args.human)
        if res is None:
            print("no judgements.csv; run `agent-eval finalize` first")
            return 1
        _print_agreement(res)
        build_report(args.campaign_dir)
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

    from .analysis.agreement import write_agreement

    finalize(d)
    agreement = write_agreement(d)  # inter-judge agreement, when the campaign has 2+ judges
    if agreement and agreement.get("pairs"):
        _print_agreement(agreement)
    res = validate(d)
    _print_validation(res)
    print("report:", build_report(d))
    return 0 if res["ok"] else 1


def _print_agreement(res: dict) -> None:
    for p in res.get("pairs", []):
        k, ac1 = p["cohen_kappa"], p["gwet_ac1"]
        lo, hi = p.get("kappa_ci", (float("nan"), float("nan")))
        print(f"judges {p['judge_a']} vs {p['judge_b']}: n={p['n']} agree={p['percent_agreement']:.1%} "
              f"kappa={k:.3f} [{lo:.3f}, {hi:.3f}] AC1={ac1:.3f} | {p['a_only']} pass only under A, "
              f"{p['b_only']} only under B (McNemar p={p['mcnemar_p']:.3f})")
    if "krippendorff_alpha" in res:
        print(f"Krippendorff alpha (all judges): {res['krippendorff_alpha']:.3f}")


def _print_validation(res: dict) -> None:
    print(f"validation: {'OK' if res['ok'] else 'FAILED'} ({res['n_errors']} errors, {res['n_warnings']} warnings)")
    for f in res["findings"][:20]:
        print(f"  [{f['severity']}] {f['check']} {f['run_id'] or ''} {f['task_id'] or ''}: {f['detail']}")


if __name__ == "__main__":
    sys.exit(main())
