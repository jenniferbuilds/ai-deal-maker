"""Automated demo showing the agent in action."""

import os
os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"

from data import load_scenarios
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler
from memory_integration import SimpleMemory, MemoryEnhancedBuyer
import random

def run_demo():
    print("="*70)
    print("NEGOTIATION AGENT DEMO - Automated Simulation")
    print("="*70)
    
    # Load scenario
    scenarios = load_scenarios("validation")
    sc = random.Random(42).choice(scenarios)
    
    print(f"\nScenario: {sc.title}")
    print(f"Category: {sc.category}")
    print(f"Listing price: {fmt(sc.listing_price)}")
    print(f"Your target: {fmt(sc.buyer_target)}")
    print(f"Secret seller floor: {fmt(sc.floor)} (unknown to agent)")
    print()
    
    # Initialize agent
    chat = ChatTokenizer(DEFAULT_BASE_MODEL)
    client = make_client()
    
    with client.session(experiment="negotiation-demo") as session:
        buyer = RiverBuyer(
            session_sampler(session, DEFAULT_BASE_MODEL),
            chat,
            temperature=0.7
        )
        
        # Add memory
        memory = SimpleMemory("demo_memory.json")
        buyer = MemoryEnhancedBuyer(buyer, memory)
        
        messages = [{"role": "system", "content": buyer_system_prompt(sc, MAX_TURNS)}]
        
        # Simulate negotiation
        seller_prices = [sc.listing_price, sc.listing_price * 0.9, sc.listing_price * 0.8, sc.floor * 1.1]
        
        print("SELLER: Hi! I'm selling this item. Asking", fmt(sc.listing_price))
        messages.append({"role": "user", "content": f"Hi! I'm selling this item. Asking {fmt(sc.listing_price)}"})
        
        for turn in range(4):
            # Buyer's turn
            gen = buyer.act([messages])[0]
            messages.append({"role": "assistant", "content": gen.text})
            text, action = parse_buyer(gen.text)
            
            print(f"BUYER: {text}")
            if action.kind == "offer":
                print(f"       [offers {fmt(action.price)}]")
            elif action.kind == "accept":
                print("       [accepts!]")
                break
            elif action.kind == "walk":
                print("       [walks away]")
                break
            
            # Seller's response
            if turn < len(seller_prices) - 1:
                seller_price = seller_prices[turn + 1]
                reply = f"I can't go that low. How about {fmt(seller_price)}?"
                print(f"SELLER: {reply}")
                messages.append({"role": "user", "content": reply})
        
        print("\n" + "="*70)
        print("Demo complete!")
        
        # Store in memory
        memory.remember({
            "category": sc.category,
            "listing_price": sc.listing_price,
            "success": action.kind in ["offer", "accept"],
            "tactics": ["lowball start", "gradual increase", "polite persistence"]
        })

if __name__ == "__main__":
    run_demo()
