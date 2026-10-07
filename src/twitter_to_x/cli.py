"""The ``twitter-to-x`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .data.anonymize import new_salt
from .sentiment import SCORERS, build_scorer


def _scorer(name: str, settings: Settings):
    if name == "transformer":
        return build_scorer(name, model_name=settings.sentiment_model, batch_size=settings.batch_size,
                            local_files_only=settings.hf_offline)
    return build_scorer(name)


def cmd_synth(args, settings) -> int:
    from .data.synthetic import make_posts

    out = Path(args.out or settings.data_dir / "raw" / "synthetic_posts.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_posts(args.n, seed=settings.seed).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


def cmd_prepare(args, settings) -> int:
    from .analysis import prepare
    from .data.sources import load_sources

    posts = load_sources(args.source, settings.salt or new_salt())
    df, manifest = prepare(posts, settings.cutoff, args.n_per_period, settings.seed)
    out = Path(args.out or settings.data_dir / "prepared")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "posts.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def cmd_analyze(args, settings) -> int:
    from .analysis import analyze, score, to_markdown, write_report

    folder = Path(args.prepared or settings.data_dir / "prepared")
    df = pd.read_csv(folder / "posts.csv", dtype={"text": str})
    scored = score(df, _scorer(args.scorer, settings))
    out = Path(args.out or settings.out_dir / args.scorer)
    out.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=["text"]).to_csv(out / "scores.csv", index=False)  # no text in the outputs
    result = analyze(scored, n_topics=args.topics, seed=settings.seed, n_boot=args.n_boot)
    write_report(result, out)
    print(to_markdown(result))
    return 0


def cmd_validate(args, settings) -> int:
    from .moderation import ModerationDetector, substring_flag
    from .moderation.validate import evaluate_flags, load_gold

    gold = load_gold(args.gold)
    for name, flag in (("word-boundary patterns", ModerationDetector().flag), ("substring baseline", substring_flag)):
        r = evaluate_flags(gold, flag)
        print(f"{name}: precision {r['precision']:.3f} {tuple(round(x, 3) for x in r['precision_ci95'])}, "
              f"recall {r['recall']:.3f} {tuple(round(x, 3) for x in r['recall_ci95'])}, F1 {r['f1']:.3f}, "
              f"false positives {r['fp']}, false negatives {r['fn']}")
    return 0


def cmd_flag(args, settings) -> int:
    from .moderation import ModerationDetector

    det = ModerationDetector()
    for text in args.text:
        m = det.match(text)
        print(json.dumps({"text": text, "moderation": m.flagged, "categories": m.categories, "terms": m.terms}))
    return 0


def cmd_demo(args, settings) -> int:
    from .analysis import analyze, prepare, score, to_markdown, write_report
    from .data.sources import load_csv
    from .data.synthetic import make_posts

    out = Path(args.out or settings.out_dir / "demo")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "synthetic_posts.csv"
    make_posts(args.n, seed=settings.seed).to_csv(raw, index=False)
    posts = load_csv(raw, new_salt(), source="synthetic")
    df, manifest = prepare(posts, settings.cutoff, args.n, settings.seed)
    print("filters:", json.dumps(manifest["filters"]))
    result = analyze(score(df, _scorer("lexicon", settings)), n_topics=args.topics, seed=settings.seed,
                     n_boot=args.n_boot)
    write_report(result, out)
    print(to_markdown(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twitter-to-x", description="Twitter vs X sentiment and moderation study.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic posts with a known truth")
    p.add_argument("--n", type=int, default=3000, help="posts per period")
    p.add_argument("--out")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("prepare", help="load, anonymize, filter and sample both periods")
    p.add_argument("--source", action="append", required=True, metavar="KIND=PATH",
                   help="sentiment140 | x_parquet | csv")
    p.add_argument("--n-per-period", type=int, default=20000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("analyze", help="score sentiment and moderation talk, run the statistics, write the report")
    p.add_argument("--scorer", choices=SCORERS, default="lexicon")
    p.add_argument("--prepared")
    p.add_argument("--out")
    p.add_argument("--topics", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=500)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("validate-moderation", help="precision and recall of the moderation patterns")
    p.add_argument("--gold", help="CSV with text,label (default: the built-in check set)")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("flag", help="show the moderation match for texts")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_flag)

    p = sub.add_parser("demo", help="offline end-to-end study on synthetic posts")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--topics", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--out")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    raise SystemExit(main())
