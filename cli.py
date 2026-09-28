#!/usr/bin/env python3
"""AI Deal Maker CLI - Quick negotiation from command line.

Usage:
    uv run python cli.py negotiate --item "iPhone 13" --price 500 --target 350
    uv run python cli.py status
    uv run python cli.py demo
"""

import argparse
import os
import json

os.environ.setdefault("RIVER_API_KEY", "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ")

def show_status():
    """Show training and system status."""
    print("\n🤖 AI DEAL MAKER - STATUS")
    print("=" * 50)
    
    try:
        with open("runs/v2/metrics.jsonl") as f:
            lines = f.readlines()
            if lines:
                latest = json.loads(lines[-1])
                print(f"\n📊 Training Complete!")
                print(f"   Steps: {latest['step'] + 1}/50")
                print(f"   Reward: {latest['mean_reward']:+.2f}")
                print(f"   Deal Rate: {latest['deal_rate']:.0%}")
                print(f"   Avg Discount: {(1 - latest['price_vs_listing']) * 100:.1f}%")
    except Exception as e:
        print(f"   Status: {e}")
    
    print("\n✅ Ready to negotiate!")
    print("=" * 50)

def quick_negotiate(item: str, price: float, target: float):
    """Quick negotiation simulation."""
    from data import Scenario
    from env import buyer_system_prompt, parse_buyer, fmt
    from policies import ChatTokenizer, RiverBuyer, make_client, session_sampler, DEFAULT_BASE_MODEL
    
    print(f"\n🤖 AI DEAL MAKER - NEGOTIATION")
    print("=" * 50)
    print(f"Item: {item}")
    print(f"Listing: ${price:,.2f}")
    print(f"Target: ${target:,.2f}")
    print("=" * 50)
    
    # Create scenario
    class QuickScenario:
        uid = "quick"
        title = item
        category = "general"
        description = f"Negotiating for {item}"
        listing_price = price
        buyer_target = target
        floor = target * 0.9
    
    sc = QuickScenario()
    
    # Initialize
    chat = ChatTokenizer(DEFAULT_BASE_MODEL)
    client = make_client()
    
    # Load checkpoint
    checkpoint = None
    try:
        with open("runs/v2/checkpoints.json") as f:
            checkpoint = json.load(f).get("latest")
    except:
        pass
    
    with client.session(experiment="cli-negotiation") as session:
        buyer = RiverBuyer(
            session_sampler(session, DEFAULT_BASE_MODEL, checkpoint),
            chat, temperature=0.7
        )
        
        messages = [{"role": "system", "content": buyer_system_prompt(sc, 6)}]
        
        # Seller opening
        seller_msg = f"Hi! I'm selling {item}. Asking ${price:,.0f}."
        print(f"\n👤 SELLER: {seller_msg}")
        messages.append({"role": "user", "content": seller_msg})
        
        # Get agent response
        gen = buyer.act([messages])[0]
        text, action = parse_buyer(gen.text)
        
        print(f"🤖 AGENT: {text}")
        if action.kind == "offer":
            print(f"   [OFFERS: ${action.price:,.2f}]")
        
        print("\n" + "=" * 50)
        print("💡 Copy the agent's message to send to the seller!")

def run_demo():
    """Run interactive demo."""
    print("\n🎯 Starting interactive demo...")
    os.system("uv run python demo_negotiation.py")

def main():
    parser = argparse.ArgumentParser(description="AI Deal Maker CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Status
    subparsers.add_parser("status", help="Show system status")
    
    # Negotiate
    neg = subparsers.add_parser("negotiate", help="Quick negotiation")
    neg.add_argument("--item", required=True, help="Item name")
    neg.add_argument("--price", type=float, required=True, help="Listing price")
    neg.add_argument("--target", type=float, required=True, help="Your target price")
    
    # Demo
    subparsers.add_parser("demo", help="Run interactive demo")
    
    args = parser.parse_args()
    
    if args.command == "status":
        show_status()
    elif args.command == "negotiate":
        quick_negotiate(args.item, args.price, args.target)
    elif args.command == "demo":
        run_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
