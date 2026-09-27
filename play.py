"""Live demo: YOU are the seller, the agent tries to buy from you.

Type replies in plain English. Commands:
  /accept   accept the buyer's latest offer
  /quit     end without a deal

    uv run python play.py --run-dir runs/v1            # trained agent
    uv run python play.py --base                       # untrained base, for contrast
"""

from __future__ import annotations

import argparse
import random

from data import load_scenarios
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from evaluate import resolve_checkpoint
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/v1")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--base", action="store_true", help="Play against the untrained base model.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    ckpt = None if args.base else resolve_checkpoint(args)
    if not args.base and not ckpt:
        raise SystemExit("No trained checkpoint found. Train first, pass --checkpoint, or use --base.")

    sc = random.Random(args.seed).choice(load_scenarios("validation"))
    print(f"\nYou're selling: {sc.title} ({sc.category})\nListed at {fmt(sc.listing_price)}."
          f"\n{sc.description}\nOpponent: {'base model' if args.base else 'trained agent'}\n")

    chat = ChatTokenizer(args.base_model)
    with make_client().session(experiment="negotiation-agent-play") as session:
        buyer = RiverBuyer(session_sampler(session, args.base_model, ckpt), chat, temperature=0.7)
        messages = [{"role": "system", "content": buyer_system_prompt(sc, args.max_turns)}]
        last_offer = None

        for _ in range(args.max_turns):
            reply = input("you (seller) > ").strip()
            if reply == "/quit":
                print("No deal.")
                return
            if reply == "/accept":
                print(f"Deal at {fmt(last_offer)}." if last_offer else "The buyer hasn't offered yet.")
                if last_offer:
                    return
                continue
            messages.append({"role": "user", "content": reply})
            gen = buyer.act([messages])[0]
            messages.append({"role": "assistant", "content": gen.text})
            text, action = parse_buyer(gen.text)
            label = {"offer": f"[offers {fmt(action.price)}]" if action.price else "",
                     "accept": "[accepts your price]", "walk": "[walks away]", "none": ""}[action.kind]
            print(f"buyer > {text} {label}\n")
            if action.kind == "offer":
                last_offer = action.price
            if action.kind in ("accept", "walk"):
                return
        print("Out of turns, no deal.")


if __name__ == "__main__":
    main()
