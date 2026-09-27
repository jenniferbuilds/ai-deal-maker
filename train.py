"""GRPO-style RL: teach a LoRA buyer to negotiate better deals, politely.

Each step:
  1. Pick K training scenarios and play G negotiations per scenario with the
     live training weights (lockstep, one batched sample call per turn).
  2. Score each negotiation with the env reward (price vs. secret floor,
     no-deal / rudeness / format penalties).
  3. Advantage = reward - mean reward of its scenario group. Every buyer turn
     in an episode gets that episode's advantage.
  4. One train_step with loss_fn="importance_sampling".

Resumable: checkpoints + progress.json live in the run dir, so rerunning
the same command continues after a crash or lost session.

    export RIVER_API_KEY=...
    uv run python train.py --run-dir runs/v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import river_client as river

from data import load_scenarios
from env import MAX_TURNS, TemplateVoice, format_transcript, run_episodes, summarize
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, LLMVoice, RiverBuyer, make_client


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/v1")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--scenarios-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--voice", choices=["template", "llm"], default="template",
                   help="How the simulated seller phrases its replies.")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- progress --

def _progress_path(run_dir: str) -> str:
    return os.path.join(run_dir, "progress.json")


def load_progress(run_dir: str) -> tuple[int, river.Checkpoint | None]:
    path = _progress_path(run_dir)
    if not os.path.exists(path):
        return 0, None
    with open(path) as f:
        p = json.load(f)
    # A Checkpoint object (not a bare path) also restores model.step.
    ckpt = river.Checkpoint(path=p["ckpt_path"], step=p["ckpt_step"], checkpoint_type="training")
    return p["next_step"], ckpt


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"  # write-then-rename so a crash can't corrupt it
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_progress(run_dir: str, next_step: int, ckpt: river.Checkpoint) -> None:
    _write_json(_progress_path(run_dir), {"next_step": next_step, "ckpt_path": ckpt.path,
                                          "ckpt_step": ckpt.step})


def record_inference_ckpt(run_dir: str, step: int, ckpt: river.Checkpoint) -> None:
    path = os.path.join(run_dir, "checkpoints.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[str(step)] = ckpt.path
    data["latest"] = ckpt.path
    _write_json(path, data)


# -------------------------------------------------------------- RL datums --

def build_datums(episodes, group_size: int) -> tuple[list[dict], int]:
    datums, skipped_groups = [], 0
    for g in range(0, len(episodes), group_size):
        group = episodes[g:g + group_size]
        rewards = [ep.reward for ep in group]
        mean_r = sum(rewards) / len(rewards)
        if all(abs(r - mean_r) < 1e-9 for r in rewards):
            skipped_groups += 1  # zero advantage -> zero gradient
            continue
        for ep, r in zip(group, rewards):
            adv = r - mean_r
            for gen in ep.generations:
                if not gen.tokens:
                    continue
                ob_len = len(gen.prompt_tokens)
                full = gen.prompt_tokens + gen.tokens
                # Pre-shifted layout: the last prompt position predicts the
                # first response token; the final slot is always masked.
                datums.append({
                    "input_ids": full,
                    "attention_mask": [1] * len(full),
                    "old_logprobs": [0.0] * (ob_len - 1) + gen.logprobs + [0.0],
                    "advantages": [0.0] * (ob_len - 1) + [adv] * len(gen.tokens) + [0.0],
                })
    return datums, skipped_groups


# -------------------------------------------------------------------- main --

def train(args, client, scenarios, chat, start_step: int, ckpt) -> None:
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    samples_path = os.path.join(args.run_dir, "train_transcripts.txt")
    k = args.scenarios_per_step

    with client.session(experiment="negotiation-agent", run=os.path.basename(args.run_dir)) as session:
        model = session.create_model(
            base_model=args.base_model,
            lora=river.LoraConfig(rank=args.rank, train_unembed=True, seed=args.seed),
            checkpoint=ckpt,
        )
        buyer = RiverBuyer(model.sample, chat, max_tokens=args.max_tokens, temperature=1.0)
        voice = (LLMVoice(session, args.base_model, chat, seed=args.seed) if args.voice == "llm"
                 else TemplateVoice(seed=args.seed))

        for step in range(start_step, args.steps):
            t0 = time.time()
            start = (step * k) % len(scenarios)
            batch = [scenarios[(start + j) % len(scenarios)] for j in range(k)]
            episodes = run_episodes([sc for sc in batch for _ in range(args.group_size)],
                                    buyer, voice, max_turns=args.max_turns)
            stats = summarize(episodes)
            datums, skipped = build_datums(episodes, args.group_size)

            row = {"step": step, **stats, "datums": len(datums), "skipped_groups": skipped}
            if datums:
                fb, opt = model.train_step(datums, lr=args.lr, loss_fn="importance_sampling",
                                           grad_clip_norm=1.0)
                row.update(kl=fb.metrics.get("kl"), mean_ratio=fb.metrics.get("mean_ratio"),
                           grad_norm=opt.metrics.get("grad_norm"))
            row["seconds"] = round(time.time() - t0, 1)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"step {step:3d} | reward {stats['mean_reward']:+.3f} | deal {stats['deal_rate']:.0%} "
                  f"| price/list {stats['price_vs_listing']:.1%} | rude {stats['rude_rate']:.0%} "
                  f"| fmt_err {stats['format_error_rate']:.0%} | datums {len(datums)} "
                  f"| kl {row.get('kl')} | {row['seconds']}s", flush=True)

            best = max(episodes, key=lambda e: e.reward)
            with open(samples_path, "a") as f:
                f.write(f"\n##### step {step} (best of batch)\n{format_transcript(best)}")

            if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
                tag = f"step_{step + 1:06d}"
                train_ckpt = model.save_weights(tag)
                save_progress(args.run_dir, step + 1, train_ckpt)
                infer_ckpt = model.save_weights(f"{tag}_infer", mode="inference")
                record_inference_ckpt(args.run_dir, step + 1, infer_ckpt)
                print(f"  saved {tag}: inference checkpoint {infer_ckpt.path}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    _write_json(os.path.join(args.run_dir, "config.json"), vars(args))

    scenarios = load_scenarios("train")
    random.Random(args.seed).shuffle(scenarios)
    print(f"{len(scenarios)} training scenarios; "
          f"{args.scenarios_per_step}x{args.group_size} negotiations per step")

    chat = ChatTokenizer(args.base_model)
    client = make_client()

    backoff = 10.0
    while True:
        start_step, ckpt = load_progress(args.run_dir)
        if start_step >= args.steps:
            print("Training already complete for this run dir.")
            break
        try:
            train(args, client, scenarios, chat, start_step, ckpt)
            break
        except (river.RiverConnectionError, river.RiverTimeoutError) as e:
            # Covers lost sessions and capacity squeezes. Resume from the last checkpoint.
            print(f"transient failure ({type(e).__name__}): {e}; resuming from step "
                  f"{load_progress(args.run_dir)[0]} in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)

    rewards = [json.loads(line)["mean_reward"]
               for line in open(os.path.join(args.run_dir, "metrics.jsonl"))]
    if len(rewards) >= 10:
        print(f"mean reward first 5 steps {statistics.mean(rewards[:5]):+.3f} -> "
              f"last 5 steps {statistics.mean(rewards[-5:]):+.3f}")


if __name__ == "__main__":
    main()
