"""Live demo: YOU are the seller, the trained agent tries to buy from you.

This uses your latest trained checkpoint from runs/v2.

Type replies in plain English. Commands:
  /accept   accept the buyer's latest offer
  /quit     end without a deal

    uv run python play_live.py
"""

from __future__ import annotations

import argparse
import random
import os

from data import load_scenarios
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler


def resolve_checkpoint(run_dir: str) -> str | None:
    """Get latest checkpoint from training run."""
    import json
    path = os.path.join(run_dir, "checkpoints.json")
    if os.path.exists(path):
        return json.load(open(path)).get("latest")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/v2")
    p.add_argument("--base", action="store_true", help="Play against untrained base model.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # Set API key if not already set
    if not os.environ.get("RIVER_API_KEY"):
        os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"

    ckpt = None if args.base else resolve_checkpoint(args.run_dir)
    if not args.base and not ckpt:
        print(f"No trained checkpoint found in {args.run_dir}.")
        print("Using base model instead. Train first or use an existing run.")
        args.base = True

    scenarios = load_scenarios("validation")
    sc = random.Random(args.seed).choice(scenarios)
    
    print("\n" + "="*60)
    print("CRAIGSLIST BUYER AGENT - LIVE NEGOTIATION")
    print("="*60)
    print(f"\nYou're selling: {sc.title}")
    print(f"Category: {sc.category}")
    print(f"Listed at: {fmt(sc.listing_price)}")
    print(f"Description: {sc.description[:100]}..." if len(sc.description) > 100 else f"Description: {sc.description}")
    print(f"\nOpponent: {'Base model (untrained)' if args.base else 'Trained negotiation agent'}")
    print("="*60)
    print("\nCommands: /accept (accept offer) | /quit (walk away)")
    print()

    chat = ChatTokenizer(args.base_model)
    with make_client().session(experiment="negotiation-agent-play") as session:
        buyer = RiverBuyer(session_sampler(session, args.base_model, ckpt), chat, temperature=0.7)
        messages = [{"role": "system", "content": buyer_system_prompt(sc, args.max_turns)}]
        last_offer = None

        # Seller's opening
        print(f"SELLER: Hi! Yes, the {sc.title} is still available. I'm asking {fmt(sc.listing_price)}.")
        messages.append({"role": "user", "content": f"Hi! Yes, the {sc.title} is still available. I'm asking {fmt(sc.listing_price)}."})

        for turn in range(args.max_turns):
            print()
            reply = input("you (seller) > ").strip()
            
            if reply.lower() == "/quit":
                print("\nSELLER: No deal. Good luck!")
                return
            if reply.lower() == "/accept":
                if last_offer:
                    print(f"\nSELLER: Deal! {fmt(last_offer)} it is. When can you pick it up?")
                    print(f"\n{'='*60}")
                    print(f"RESULT: DEAL at {fmt(last_offer)}")
                    print(f"Listing price was: {fmt(sc.listing_price)}")
                    print(f"Buyer saved: {fmt(sc.listing_price - last_offer)} ({(sc.listing_price - last_offer)/sc.listing_price*100:.1f}% off)")
                    return
                else:
                    print("The buyer hasn't made an offer yet. Make them an offer first!")
                    continue
            
            messages.append({"role": "user", "content": reply})
            
            try:
                gen = buyer.act([messages])[0]
            except Exception as e:
                print(f"Error: {e}")
                continue
                
            messages.append({"role": "assistant", "content": gen.text})
            text, action = parse_buyer(gen.text)
            
            label = {
                "offer": f"[offers {fmt(action.price)}]" if action.price else "",
                "accept": "[accepts your price]",
                "walk": "[walks away]",
                "none": ""
            }[action.kind]
            
            print(f"BUYER: {text} {label}")
            
            if action.kind == "offer":
                last_offer = action.price
            if action.kind == "accept":
                print(f"\n{'='*60}")
                print(f"RESULT: DEAL at {fmt(sc.listing_price)}")
                return
            if action.kind == "walk":
                print(f"\n{'='*60}")
                print("RESULT: NO DEAL - Buyer walked away")
                return
        
        print(f"\n{'='*60}")
        print("RESULT: NO DEAL - Out of turns")

if __name__ == "__main__":
    main()
