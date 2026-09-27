"""Unified negotiation agent with:
1. Multi-platform support (Craigslist, eBay, salary, rent)
2. Procedural memory (via Memorable or file-based fallback)
3. RL-trained negotiation strategies (via River)
4. Live play against humans

Usage:
    # Train on Craigslist data (default)
    uv run python agent.py train --platform craigslist
    
    # Play live negotiation
    uv run python agent.py play --platform ebay
    
    # Use memory-enhanced agent
    uv run python agent.py play --platform salary --use-memory
"""

from __future__ import annotations

import argparse
import json
import os
import random

from data import load_scenarios
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler
from multi_platform import (
    NegotiationScenario, NegotiationType, get_adapter,
    ECOMMERCE_SCENARIO, SALARY_SCENARIO, RENT_SCENARIO
)
from memory_integration import MemoryEnhancedBuyer, NegotiationMemory, SimpleMemory


def train_agent(args):
    """Train the agent on specified platform."""
    print(f"Training on {args.platform} platform...")
    
    if args.platform == "craigslist":
        # Use existing training script
        import train
        train.main()
    else:
        print(f"Training for {args.platform} not yet implemented.")
        print("Platform-specific datasets needed.")


def play_live(args):
    """Live negotiation demo."""
    # Set API key
    if not os.environ.get("RIVER_API_KEY"):
        os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"
    
    # Select scenario based on platform
    platform_scenarios = {
        "craigslist": ("validation", None),  # Use Craigslist data
        "ebay": (None, ECOMMERCE_SCENARIO),
        "salary": (None, SALARY_SCENARIO),
        "rent": (None, RENT_SCENARIO),
    }
    
    split, custom_scenario = platform_scenarios.get(args.platform, ("validation", None))
    
    if custom_scenario:
        scenario = custom_scenario
    else:
        scenarios = load_scenarios(split)
        scenario = random.Random(args.seed).choice(scenarios)
    
    # Get platform adapter
    negotiation_type = {
        "craigslist": NegotiationType.PRICE_ONLY,
        "ebay": NegotiationType.PRICE_ONLY,
        "salary": NegotiationType.SALARY,
        "rent": NegotiationType.RENT,
    }.get(args.platform, NegotiationType.PRICE_ONLY)
    
    adapter = get_adapter(negotiation_type)
    
    # Load checkpoint
    ckpt = None
    if not args.base:
        ckpt_path = os.path.join(args.run_dir, "checkpoints.json")
        if os.path.exists(ckpt_path):
            ckpt = json.load(open(ckpt_path)).get("latest")
            print(f"Using trained checkpoint: {ckpt}")
        else:
            print(f"No checkpoint found in {args.run_dir}, using base model")
    
    print("\n" + "="*60)
    print(f"{args.platform.upper()} NEGOTIATION AGENT - LIVE DEMO")
    print("="*60)
    
    if hasattr(scenario, 'title'):
        print(f"\nItem: {scenario.title}")
        print(f"Category: {scenario.category}")
        print(f"Listed at: {fmt(scenario.listing_price)}")
        print(f"Description: {scenario.description[:100]}...")
    else:
        print(f"\nScenario: {scenario.description}")
        print(f"Your target: ${scenario.your_target:,.2f}")
        print(f"Context: {scenario.context}")
    
    print(f"\nOpponent: {'Base model (untrained)' if args.base else 'Trained agent'}")
    print(f"Memory: {'Enabled' if args.use_memory else 'Disabled'}")
    print("="*60)
    print("\nCommands: /accept | /quit | /offer <amount>")
    print()
    
    # Initialize buyer
    chat = ChatTokenizer(args.base_model)
    client = make_client()
    
    with client.session(experiment=f"negotiation-agent-{args.platform}") as session:
        buyer = RiverBuyer(
            session_sampler(session, args.base_model, ckpt),
            chat,
            temperature=args.temperature
        )
        
        # Wrap with memory if requested
        if args.use_memory:
            memory = SimpleMemory(f"{args.platform}_memory.json")
            buyer = MemoryEnhancedBuyer(buyer, memory)
        
        # Build initial prompt
        if hasattr(scenario, 'title'):
            system_prompt = buyer_system_prompt(scenario, args.max_turns)
        else:
            system_prompt = f"You are negotiating {scenario.description}. "
            system_prompt += f"Your target: ${scenario.your_target:,.2f}. "
            system_prompt += f"Context: {scenario.context}. "
            system_prompt += "Negotiate the best deal. "
            system_prompt += "End EVERY message with: ACTION: OFFER <amount> or ACTION: ACCEPT or ACTION: WALK"
        
        messages = [{"role": "system", "content": system_prompt}]
        last_offer = None
        
        # Your opening as seller
        if hasattr(scenario, 'listing_price'):
            opening = f"Hi! I'm selling {scenario.title}. Asking {fmt(scenario.listing_price)}."
        else:
            opening = f"Hi! I saw your interest. My asking price is ${scenario.your_target * 1.2:,.2f}."
        
        print(f"SELLER: {opening}")
        messages.append({"role": "user", "content": opening})
        
        for turn in range(args.max_turns):
            print()
            reply = input("you (seller) > ").strip()
            
            if reply.lower() == "/quit":
                print("\nSELLER: No deal. Thanks for your time!")
                return
            
            if reply.lower() == "/accept":
                if last_offer:
                    print(f"\nSELLER: Deal! Agreed at ${last_offer:,.2f}.")
                    print(f"\n{'='*60}")
                    print(f"RESULT: DEAL at ${last_offer:,.2f}")
                    if hasattr(scenario, 'listing_price'):
                        print(f"Original price: {fmt(scenario.listing_price)}")
                        print(f"Discount: {fmt(scenario.listing_price - last_offer)} ({(scenario.listing_price - last_offer)/scenario.listing_price*100:.1f}%)")
                    return
                else:
                    print("No offer to accept yet!")
                    continue
            
            if reply.lower().startswith("/offer "):
                try:
                    amount = float(reply.split()[1])
                    print(f"\nSELLER: I can offer ${amount:,.2f}.")
                    reply = f"I can do ${amount:,.2f}."
                except:
                    print("Invalid offer format. Use: /offer <amount>")
                    continue
            
            messages.append({"role": "user", "content": reply})
            
            try:
                gen = buyer.act([messages])[0]
                messages.append({"role": "assistant", "content": gen.text})
                
                # Parse response
                text, action = parse_buyer(gen.text)
                
                label = {
                    "offer": f"[offers ${action.price:,.2f}]" if action.price else "",
                    "accept": "[accepts]",
                    "walk": "[walks away]",
                    "none": ""
                }[action.kind]
                
                print(f"BUYER: {text} {label}")
                
                if action.kind == "offer":
                    last_offer = action.price
                elif action.kind == "accept":
                    print(f"\n{'='*60}")
                    print("RESULT: DEAL!")
                    return
                elif action.kind == "walk":
                    print(f"\n{'='*60}")
                    print("RESULT: NO DEAL - Buyer walked away")
                    return
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("RESULT: NO DEAL - Out of turns")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-platform negotiation agent with memory",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--platform", choices=["craigslist", "ebay", "salary", "rent"],
                             default="craigslist", help="Platform to train for")
    train_parser.add_argument("--run-dir", default="runs/v2")
    train_parser.add_argument("--steps", type=int, default=50)
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play live negotiation")
    play_parser.add_argument("--platform", choices=["craigslist", "ebay", "salary", "rent"],
                            default="craigslist", help="Negotiation platform")
    play_parser.add_argument("--run-dir", default="runs/v2")
    play_parser.add_argument("--base", action="store_true", help="Use untrained base model")
    play_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    play_parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    play_parser.add_argument("--temperature", type=float, default=0.7)
    play_parser.add_argument("--seed", type=int, default=None)
    play_parser.add_argument("--use-memory", action="store_true", help="Enable procedural memory")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_agent(args)
    elif args.command == "play":
        play_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
