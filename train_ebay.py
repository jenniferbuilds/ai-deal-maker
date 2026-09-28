"""Train negotiation agent on eBay Best Offer data.

Combines:
1. Craigslist Bargains dataset (existing)
2. eBay synthetic scenarios (new)
3. Job negotiation patterns (future)

    export RIVER_API_KEY=...
    uv run python train_ebay.py --run-dir runs/ebay_v1 --steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass

import river_client as river

from data import load_scenarios as load_craigslist
from ebay_data import generate_ebay_scenarios, EbayScenario
from env import MAX_TURNS, TemplateVoice, format_transcript, run_episodes, summarize
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, LLMVoice, RiverBuyer, make_client


@dataclass(frozen=True)
class UnifiedScenario:
    """Unified scenario format for multi-platform training."""
    uid: str
    title: str
    category: str
    description: str
    listing_price: float
    buyer_target: float
    floor: float
    platform: str
    human_prices: tuple = ()


def load_unified_scenarios(split: str = "train", include_ebay: bool = True) -> list[UnifiedScenario]:
    """Load scenarios from multiple sources."""
    scenarios = []
    
    # Load Craigslist data
    craigslist = load_craigslist(split)
    for sc in craigslist:
        scenarios.append(UnifiedScenario(
            uid=sc.uid,
            title=sc.title,
            category=sc.category,
            description=sc.description,
            listing_price=sc.listing_price,
            buyer_target=sc.buyer_target,
            floor=sc.floor,
            platform="craigslist",
            human_prices=sc.human_prices,
        ))
    
    # Add eBay scenarios
    if include_ebay:
        ebay = generate_ebay_scenarios(n=1000, seed=42 if split == "train" else 123)
        for sc in ebay:
            scenarios.append(UnifiedScenario(
                uid=sc.uid,
                title=sc.title,
                category=sc.category,
                description=sc.description,
                listing_price=sc.listing_price,
                buyer_target=sc.buyer_target,
                floor=sc.seller_floor,
                platform="ebay",
            ))
    
    print(f"Loaded {len(scenarios)} total scenarios ({len(craigslist)} Craigslist + {len(scenarios) - len(craigslist)} eBay)")
    return scenarios


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/ebay_v1")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--scenarios-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--voice", choices=["template", "llm"], default="template")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", default=None, help="Resume from existing checkpoint")
    return p.parse_args()


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_datums(episodes, group_size: int) -> tuple[list[dict], int]:
    """Build training datums from episodes."""
    datums, skipped_groups = [], 0
    for g in range(0, len(episodes), group_size):
        group = episodes[g:g + group_size]
        rewards = [ep.reward for ep in group]
        mean_r = sum(rewards) / len(rewards)
        if all(abs(r - mean_r) < 1e-9 for r in rewards):
            skipped_groups += 1
            continue
        for ep, r in zip(group, rewards):
            adv = r - mean_r
            for gen in ep.generations:
                if not gen.tokens:
                    continue
                ob_len = len(gen.prompt_tokens)
                full = gen.prompt_tokens + gen.tokens
                datums.append({
                    "input_ids": full,
                    "attention_mask": [1] * len(full),
                    "old_logprobs": [0.0] * (ob_len - 1) + gen.logprobs + [0.0],
                    "advantages": [0.0] * (ob_len - 1) + [adv] * len(gen.tokens) + [0.0],
                })
    return datums, skipped_groups


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    _write_json(os.path.join(args.run_dir, "config.json"), vars(args))

    # Load unified scenarios
    scenarios = load_unified_scenarios("train", include_ebay=True)
    random.Random(args.seed).shuffle(scenarios)
    
    # Platform distribution
    platforms = {}
    for sc in scenarios:
        platforms[sc.platform] = platforms.get(sc.platform, 0) + 1
    print(f"Platform distribution: {platforms}")
    print(f"{args.scenarios_per_step}x{args.group_size} negotiations per step")

    chat = ChatTokenizer(args.base_model)
    client = make_client()
    
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    samples_path = os.path.join(args.run_dir, "train_transcripts.txt")
    k = args.scenarios_per_step

    with client.session(experiment="negotiation-agent-ebay", run=os.path.basename(args.run_dir)) as session:
        # Load checkpoint if resuming
        ckpt = None
        if args.resume_from:
            ckpt = river.Checkpoint(path=args.resume_from, step=0, checkpoint_type="training")
            print(f"Resuming from checkpoint: {args.resume_from}")
        
        model = session.create_model(
            base_model=args.base_model,
            lora=river.LoraConfig(rank=args.rank, train_unembed=True, seed=args.seed),
            checkpoint=ckpt,
        )
        buyer = RiverBuyer(model.sample, chat, max_tokens=args.max_tokens, temperature=1.0)
        voice = (LLMVoice(session, args.base_model, chat, seed=args.seed) if args.voice == "llm"
                 else TemplateVoice(seed=args.seed))

        for step in range(args.steps):
            t0 = time.time()
            start = (step * k) % len(scenarios)
            batch = [scenarios[(start + j) % len(scenarios)] for j in range(k)]
            
            # Track platform distribution in batch
            batch_platforms = {}
            for sc in batch:
                batch_platforms[sc.platform] = batch_platforms.get(sc.platform, 0) + 1
            
            episodes = run_episodes([sc for sc in batch for _ in range(args.group_size)],
                                    buyer, voice, max_turns=args.max_turns)
            stats = summarize(episodes)
            datums, skipped = build_datums(episodes, args.group_size)

            row = {"step": step, "platforms": batch_platforms, **stats, "datums": len(datums), "skipped_groups": skipped}
            if datums:
                fb, opt = model.train_step(datums, lr=args.lr, loss_fn="importance_sampling",
                                           grad_clip_norm=1.0)
                row.update(kl=fb.metrics.get("kl"), mean_ratio=fb.metrics.get("mean_ratio"),
                           grad_norm=opt.metrics.get("grad_norm"))
            row["seconds"] = round(time.time() - t0, 1)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"step {step:3d} | reward {stats['mean_reward']:+.3f} | deal {stats['deal_rate']:.0%} "
                  f"| price/list {stats['price_vs_listing']:.1%} | platforms {batch_platforms} "
                  f"| {row['seconds']}s", flush=True)

            best = max(episodes, key=lambda e: e.reward)
            with open(samples_path, "a") as f:
                f.write(f"\n##### step {step} (best of batch)\n{format_transcript(best)}")

            if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
                tag = f"step_{step + 1:06d}"
                train_ckpt = model.save_weights(tag)
                infer_ckpt = model.save_weights(f"{tag}_infer", mode="inference")
                
                # Save checkpoint info
                ckpt_path = os.path.join(args.run_dir, "checkpoints.json")
                ckpt_data = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}
                ckpt_data[str(step + 1)] = infer_ckpt.path
                ckpt_data["latest"] = infer_ckpt.path
                _write_json(ckpt_path, ckpt_data)
                
                print(f"  saved {tag}: {infer_ckpt.path}", flush=True)

    # Final summary
    rewards = [json.loads(line)["mean_reward"]
               for line in open(metrics_path)]
    if len(rewards) >= 5:
        print(f"\n📊 Training complete!")
        print(f"   First 5 steps avg reward: {statistics.mean(rewards[:5]):+.3f}")
        print(f"   Last 5 steps avg reward: {statistics.mean(rewards[-5:]):+.3f}")


if __name__ == "__main__":
    main()
