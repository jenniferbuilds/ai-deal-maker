"""Compare buyers on held-out (validation) scenarios the agent never trained on.

Buyers:
  base      - untrained base model via River
  trained   - your RL checkpoint via River (skipped if none found)
  naive     - scripted pushover (offline reference)
  anchor    - scripted "anchor low, concede slowly" strategy (offline reference)
Also prints what real humans paid on the same listings (they faced human
sellers, so treat it as context, not a head-to-head).

    uv run python evaluate.py --run-dir runs/v1          # base vs trained
    uv run python evaluate.py --dry-run                  # offline, no API key
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from data import load_scenarios
from env import MAX_TURNS, TemplateVoice, format_transcript, run_episodes, summarize
from policies import (DEFAULT_BASE_MODEL, ChatTokenizer, LLMVoice, RiverBuyer, ScriptedBuyer,
                      make_client, session_sampler)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/v1")
    p.add_argument("--checkpoint", default=None,
                   help="Inference checkpoint path. Defaults to 'latest' in <run-dir>/checkpoints.json.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("-n", "--num-scenarios", type=int, default=60)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--voice", choices=["template", "llm"], default="template")
    p.add_argument("--dry-run", action="store_true", help="Scripted buyers only; no River calls.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def resolve_checkpoint(args) -> str | None:
    if args.checkpoint:
        return args.checkpoint
    path = os.path.join(args.run_dir, "checkpoints.json")
    if os.path.exists(path):
        return json.load(open(path)).get("latest")
    return None


def human_reference(scenarios) -> dict[str, float]:
    with_deal = [s for s in scenarios if s.human_prices]
    ratios = [p / s.listing_price for s in with_deal for p in s.human_prices]
    return {"listings_with_human_deal": len(with_deal),
            "price_vs_listing": sum(ratios) / len(ratios) if ratios else float("nan")}


def print_table(results: dict[str, dict]) -> None:
    cols = [("mean_reward", "reward", "{:+.3f}"), ("deal_rate", "deal", "{:.0%}"),
            ("price_vs_listing", "price/list", "{:.1%}"), ("price_vs_floor", "price/floor", "{:.2f}x"),
            ("rude_rate", "rude", "{:.0%}"), ("format_error_rate", "fmt_err", "{:.0%}"),
            ("mean_buyer_turns", "turns", "{:.1f}")]
    print(f"\n{'buyer':<10}" + "".join(f"{h:>13}" for _, h, _ in cols))
    for name, stats in results.items():
        cells = []
        for key, _, f in cols:
            v = stats.get(key, float("nan"))
            cells.append(f"{'-' if isinstance(v, float) and math.isnan(v) else f.format(v):>13}")
        print(f"{name:<10}" + "".join(cells))


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    scenarios = load_scenarios("validation")
    random.Random(args.seed).shuffle(scenarios)
    scenarios = scenarios[: args.num_scenarios]
    print(f"Evaluating on {len(scenarios)} held-out validation scenarios")

    buyers = {"naive": ScriptedBuyer(naive=True, seed=args.seed),
              "anchor": ScriptedBuyer(naive=False, seed=args.seed)}
    session_ctx = None
    voice = TemplateVoice(seed=args.seed)

    if not args.dry_run:
        chat = ChatTokenizer(args.base_model)
        session_ctx = make_client().session(experiment="negotiation-agent-eval")
        session = session_ctx.__enter__()
        if args.voice == "llm":
            voice = LLMVoice(session, args.base_model, chat, seed=args.seed)
        buyers = {"base": RiverBuyer(session_sampler(session, args.base_model), chat,
                                     temperature=args.temperature), **buyers}
        ckpt = resolve_checkpoint(args)
        if ckpt:
            print(f"Trained checkpoint: {ckpt}")
            buyers = {"base": buyers.pop("base"),
                      "trained": RiverBuyer(session_sampler(session, args.base_model, ckpt), chat,
                                            temperature=args.temperature), **buyers}
        else:
            print("No trained checkpoint found; evaluating base model only.")

    results, transcripts = {}, []
    try:
        for name, buyer in buyers.items():
            # Same voice seed per buyer so everyone hears the same seller phrasing.
            if isinstance(voice, TemplateVoice):
                voice = TemplateVoice(seed=args.seed)
            episodes = run_episodes(scenarios, buyer, voice, max_turns=args.max_turns)
            results[name] = summarize(episodes)
            transcripts.append((name, episodes))
            print(f"  {name}: mean reward {results[name]['mean_reward']:+.3f}")
    finally:
        if session_ctx is not None:
            session_ctx.__exit__(None, None, None)

    print_table(results)
    ref = human_reference(scenarios)
    print(f"\nHumans on these listings (vs. human sellers): paid {ref['price_vs_listing']:.1%} "
          f"of listing on {ref['listings_with_human_deal']} deals")

    tag = "dry_run" if args.dry_run else "eval"
    with open(os.path.join(args.run_dir, f"{tag}.json"), "w") as f:
        json.dump({"results": results, "human_reference": ref}, f, indent=2)
    # Side-by-side transcripts for the first few scenarios: great demo material.
    with open(os.path.join(args.run_dir, f"{tag}_transcripts.txt"), "w") as f:
        for i in range(min(8, len(scenarios))):
            for name, episodes in transcripts:
                f.write(f"[{name}] {format_transcript(episodes[i])}\n")
            f.write("\n")
    print(f"Wrote {args.run_dir}/{tag}.json and {tag}_transcripts.txt")


if __name__ == "__main__":
    main()
