"""Interactive negotiation demo for recording."""

import os
os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"

from data import load_scenarios, Scenario
from env import MAX_TURNS, buyer_system_prompt, fmt, parse_buyer
from policies import DEFAULT_BASE_MODEL, ChatTokenizer, RiverBuyer, make_client, session_sampler
import json
import time

def create_custom_scenario():
    """Create a custom test scenario."""
    from dataclasses import dataclass
    
    @dataclass(frozen=True)
    class CustomScenario:
        uid: str
        title: str
        category: str
        description: str
        listing_price: float
        buyer_target: float
        floor: float
        
    return CustomScenario(
        uid="test_001",
        title="Vintage Camera - Canon AE-1 with 50mm Lens",
        category="electronics",
        description="Excellent condition Canon AE-1 35mm film camera with 50mm f/1.8 lens. Recently serviced, light meter works perfectly. Includes original leather case and strap.",
        listing_price=250.0,
        buyer_target=180.0,
        floor=175.0
    )

def run_demo():
    print("\n" + "="*80)
    print("🤖 AI NEGOTIATION AGENT - LIVE DEMONSTRATION")
    print("="*80)
    print("\n📹 Recording started...")
    print()
    
    # Use custom scenario
    sc = create_custom_scenario()
    
    print("📋 SCENARIO DETAILS")
    print("-" * 80)
    print(f"Item:          {sc.title}")
    print(f"Category:      {sc.category}")
    print(f"Listing Price: {fmt(sc.listing_price)}")
    print(f"Buyer Target:  {fmt(sc.buyer_target)}")
    print(f"Seller Floor:  {fmt(sc.floor)} (hidden from agent)")
    print(f"Description:   {sc.description}")
    print("-" * 80)
    print()
    
    # Initialize agent
    print("🔧 Loading trained negotiation agent...")
    chat = ChatTokenizer(DEFAULT_BASE_MODEL)
    client = make_client()
    
    # Load checkpoint
    checkpoint = None
    try:
        with open("runs/v2/checkpoints.json") as f:
            checkpoints = json.load(f)
            checkpoint = checkpoints.get("latest")
            if checkpoint:
                print(f"✅ Loaded checkpoint from training run")
    except:
        print("⚠️  Using base model (untrained)")
    
    print()
    time.sleep(1)
    
    # Start negotiation
    with client.session(experiment="demo-recording") as session:
        buyer = RiverBuyer(
            session_sampler(session, DEFAULT_BASE_MODEL, checkpoint),
            chat,
            temperature=0.7
        )
        
        # Build conversation
        messages = [{"role": "system", "content": buyer_system_prompt(sc, MAX_TURNS)}]
        
        print("="*80)
        print("🎯 NEGOTIATION BEGINS")
        print("="*80)
        print()
        
        # Seller opening
        seller_msg = f"Hi! Thanks for your interest in the Canon AE-1. It's in great shape. I'm asking {fmt(sc.listing_price)}."
        print(f"👤 SELLER: {seller_msg}")
        print()
        messages.append({"role": "user", "content": seller_msg})
        
        # Track offers
        buyer_offers = []
        seller_offers = [sc.listing_price]
        deal_price = None
        
        # Negotiation loop
        for turn in range(MAX_TURNS):
            # Buyer's turn
            gen = buyer.act([messages])[0]
            messages.append({"role": "assistant", "content": gen.text})
            text, action = parse_buyer(gen.text)
            
            # Display buyer response
            if action.kind == "offer":
                buyer_offers.append(action.price)
                print(f"🤖 AGENT: {text}")
                print(f"          [OFFER: {fmt(action.price)}]")
                print()
                
                # Seller response logic
                if action.price >= sc.floor:
                    # Accept if at or above floor
                    deal_price = action.price
                    print(f"👤 SELLER: Deal! {fmt(action.price)} works for me. Let's meet up tomorrow?")
                    print()
                    print("="*80)
                    print("🎉 SUCCESS! DEAL REACHED!")
                    print("="*80)
                    print(f"  Final Price:  {fmt(deal_price)}")
                    print(f"  Original:     {fmt(sc.listing_price)}")
                    print(f"  You Saved:    {fmt(sc.listing_price - deal_price)} ({(sc.listing_price - deal_price)/sc.listing_price*100:.1f}%)")
                    print(f"  Your Target:  {fmt(sc.buyer_target)}")
                    if deal_price <= sc.buyer_target:
                        print(f"  ✅ Beat your target by {fmt(sc.buyer_target - deal_price)}!")
                    print("="*80)
                    return
                    
            elif action.kind == "accept":
                deal_price = seller_offers[-1]
                print(f"🤖 AGENT: {text}")
                print(f"          [ACCEPTS]")
                print()
                print("="*80)
                print("🎉 DEAL REACHED!")
                print("="*80)
                print(f"  Final Price: {fmt(deal_price)}")
                print(f"  You Saved:    {fmt(sc.listing_price - deal_price)} ({(sc.listing_price - deal_price)/sc.listing_price*100:.1f}%)")
                print("="*80)
                return
                
            elif action.kind == "walk":
                print(f"🤖 AGENT: {text}")
                print(f"          [WALKS AWAY]")
                print()
                print("="*80)
                print("❌ NO DEAL - Agent walked away")
                print("="*80)
                return
            
            time.sleep(1.5)
            
            # Seller counter-offer
            if turn < MAX_TURNS - 1:
                # Gradual price reduction
                last_buyer_offer = buyer_offers[-1] if buyer_offers else sc.listing_price * 0.5
                seller_counter = max(sc.floor, min(seller_offers[-1] * 0.92, sc.listing_price * 0.85))
                
                # Make it realistic
                if last_buyer_offer < sc.floor:
                    gap = seller_offers[-1] - last_buyer_offer
                    seller_counter = max(sc.floor, seller_offers[-1] - gap * 0.3)
                
                seller_offers.append(seller_counter)
                
                seller_msg = f"I appreciate the offer, but that's a bit low for this condition. I could do {fmt(seller_counter)}."
                print(f"👤 SELLER: {seller_msg}")
                print()
                messages.append({"role": "user", "content": seller_msg})
        
        print("="*80)
        print("⏱️  OUT OF TURNS - Negotiation ended without deal")
        print("="*80)

if __name__ == "__main__":
    run_demo()
