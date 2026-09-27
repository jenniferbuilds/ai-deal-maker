"""Test scenario for negotiation agent demonstration."""

import os
os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"

from data import load_scenarios
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler
import json
import time

def run_test():
    print("="*80)
    print("🤖 NEGOTIATION AGENT TEST & DEMONSTRATION")
    print("="*80)
    print()
    
    # Load a specific scenario
    scenarios = load_scenarios("validation")
    import random
    random.seed(12345)
    sc = random.choice(scenarios)
    
    print("📋 TEST SCENARIO")
    print(f"Item: {sc.title}")
    print(f"Category: {sc.category}")
    print(f"Listing Price: {fmt(sc.listing_price)}")
    print(f"Buyer Target: {fmt(sc.buyer_target)}")
    print(f"Secret Seller Floor: {fmt(sc.floor)} (agent doesn't know this)")
    print(f"Description: {sc.description[:150]}...")
    print()
    
    # Initialize agent
    print("🔧 Initializing negotiation agent...")
    chat = ChatTokenizer(DEFAULT_BASE_MODEL)
    client = make_client()
    
    # Use trained checkpoint if available
    checkpoint = None
    try:
        with open("runs/v2/checkpoints.json") as f:
            checkpoints = json.load(f)
            checkpoint = checkpoints.get("latest")
            if checkpoint:
                print(f"✅ Using trained checkpoint: {checkpoint[:50]}...")
    except:
        print("⚠️  No trained checkpoint found, using base model")
    
    print()
    
    # Run negotiation
    with client.session(experiment="test-negotiation") as session:
        buyer = RiverBuyer(
            session_sampler(session, DEFAULT_BASE_MODEL, checkpoint),
            chat,
            temperature=0.7
        )
        
        messages = [{"role": "system", "content": buyer_system_prompt(sc, MAX_TURNS)}]
        
        print("="*80)
        print("🎭 NEGOTIATION SIMULATION")
        print("="*80)
        print()
        
        # Seller opening
        seller_msg = f"Hi! I'm selling {sc.title}. Asking {fmt(sc.listing_price)}."
        print(f"SELLER: {seller_msg}")
        messages.append({"role": "user", "content": seller_msg})
        
        # Simulate negotiation
        seller_prices = [sc.listing_price, sc.listing_price * 0.85, sc.floor * 1.15, sc.floor]
        
        for turn in range(MAX_TURNS):
            # Buyer's turn
            gen = buyer.act([messages])[0]
            messages.append({"role": "assistant", "content": gen.text})
            text, action = parse_buyer(gen.text)
            
            print(f"BUYER: {text}")
            if action.kind == "offer":
                print(f"       [ACTION: OFFER {fmt(action.price)}]")
            elif action.kind == "accept":
                print(f"       [ACTION: ACCEPT]")
                print()
                print("="*80)
                print("✅ DEAL REACHED!")
                print(f"Final Price: {fmt(sc.listing_price)}")
                print(f"Original Price: {fmt(sc.listing_price)}")
                print(f"Savings: {fmt(sc.listing_price - sc.listing_price)} (0%)")
                return
            elif action.kind == "walk":
                print(f"       [ACTION: WALK]")
                print()
                print("="*80)
                print("❌ NO DEAL - Buyer walked away")
                return
            
            time.sleep(1)  # Dramatic pause
            
            # Seller response (simulated)
            if turn < len(seller_prices) - 1:
                next_price = seller_prices[turn + 1]
                seller_msg = f"I can't go that low. Best I can do is {fmt(next_price)}."
                print(f"SELLER: {seller_msg}")
                messages.append({"role": "user", "content": seller_msg})
        
        print()
        print("="*80)
        print("⏱️  OUT OF TURNS - No deal reached")

if __name__ == "__main__":
    run_test()
