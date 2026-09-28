"""eBay Best Offer negotiation data loader.

Sources:
- NBER eBay Best Offer Dataset (25M+ negotiations)
- HuggingFace negotiation datasets
- Synthetic eBay-style scenarios

This module creates eBay-specific training scenarios based on
real bargaining patterns from academic research.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Optional
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ebay_data")

# eBay categories with typical price ranges and discount patterns
EBAY_CATEGORIES = {
    "electronics": {
        "items": [
            ("iPhone 13 Pro 256GB", 600, 900),
            ("MacBook Pro M2 14-inch", 1200, 1800),
            ("Sony WH-1000XM5 Headphones", 250, 350),
            ("Nintendo Switch OLED", 280, 350),
            ("iPad Air 5th Gen", 400, 550),
            ("Canon EOS R6 Camera", 1500, 2200),
            ("DJI Mini 3 Pro Drone", 650, 900),
            ("Samsung Galaxy S23 Ultra", 700, 1000),
            ("Apple Watch Series 8", 300, 450),
            ("Bose QuietComfort Earbuds", 180, 280),
        ],
        "avg_discount": 0.15,
        "negotiation_style": "technical",
    },
    "fashion": {
        "items": [
            ("Gucci GG Marmont Bag", 1500, 2500),
            ("Louis Vuitton Neverfull MM", 1200, 1800),
            ("Nike Air Jordan 1 Retro", 150, 250),
            ("Rolex Submariner Watch", 8000, 12000),
            ("Chanel Classic Flap Bag", 5000, 8000),
            ("Vintage Levi's 501 Jeans", 80, 150),
            ("Burberry Trench Coat", 800, 1400),
            ("Hermes Silk Scarf", 300, 500),
            ("Ray-Ban Aviator Sunglasses", 100, 180),
            ("Canada Goose Parka", 600, 1000),
        ],
        "avg_discount": 0.20,
        "negotiation_style": "luxury",
    },
    "collectibles": {
        "items": [
            ("Pokemon Base Set Charizard PSA 9", 300, 500),
            ("1st Edition Magic Card", 200, 400),
            ("Vintage Star Wars Figure", 100, 200),
            ("Signed Michael Jordan Jersey", 500, 1000),
            ("Rare Vinyl Record Collection", 150, 300),
            ("Antique Pocket Watch", 200, 400),
            ("Comic Book CGC 9.8", 250, 500),
            ("Sports Memorabilia Lot", 300, 600),
            ("Vintage Concert Poster", 100, 250),
            ("Limited Edition Sneakers", 400, 700),
        ],
        "avg_discount": 0.25,
        "negotiation_style": "collector",
    },
    "home_garden": {
        "items": [
            ("KitchenAid Stand Mixer", 250, 400),
            ("Dyson V15 Vacuum", 450, 650),
            ("Herman Miller Aeron Chair", 800, 1200),
            ("Weber Genesis Grill", 600, 900),
            ("Vitamix Blender", 300, 500),
            ("Breville Espresso Machine", 500, 800),
            ("Roomba j7+", 400, 600),
            ("All-Clad Cookware Set", 600, 900),
            ("Casper Mattress Queen", 700, 1100),
            ("Sonos Home Theater System", 800, 1200),
        ],
        "avg_discount": 0.18,
        "negotiation_style": "practical",
    },
    "automotive": {
        "items": [
            ("Car Wheels Set of 4", 400, 800),
            ("Performance Exhaust System", 500, 900),
            ("Leather Seat Covers", 150, 300),
            ("Car Audio System", 300, 600),
            ("OEM Brake Kit", 200, 400),
            ("Roof Rack System", 250, 450),
            ("LED Headlight Kit", 150, 300),
            ("Tonneau Cover", 300, 500),
            ("Performance Air Intake", 200, 400),
            ("Winch 12000 lb", 350, 600),
        ],
        "avg_discount": 0.15,
        "negotiation_style": "parts",
    },
}

# eBay-specific negotiation patterns from research
EBAY_PATTERNS = {
    "first_offer_ratio": (0.70, 0.85),  # Buyers typically offer 70-85% of asking
    "counter_ratio": (0.90, 0.95),  # Sellers counter at 90-95% of asking
    "final_deal_ratio": (0.82, 0.92),  # Deals close at 82-92% of asking
    "max_rounds": 3,  # eBay allows up to 3 rounds
    "acceptance_threshold": 0.88,  # Sellers often accept at 88%+ of asking
}


@dataclass(frozen=True)
class EbayScenario:
    """An eBay Best Offer negotiation scenario."""
    uid: str
    title: str
    category: str
    condition: str
    listing_price: float
    buyer_target: float
    seller_floor: float  # Minimum seller will accept
    item_cost: float  # What seller paid (their break-even)
    shipping: float
    description: str
    negotiation_style: str


def generate_ebay_scenarios(n: int = 1000, seed: int = 42) -> list[EbayScenario]:
    """Generate synthetic eBay negotiation scenarios based on real patterns."""
    rng = random.Random(seed)
    scenarios = []
    
    for i in range(n):
        # Pick random category
        category = rng.choice(list(EBAY_CATEGORIES.keys()))
        cat_data = EBAY_CATEGORIES[category]
        
        # Pick random item
        item_name, price_low, price_high = rng.choice(cat_data["items"])
        
        # Generate prices based on eBay patterns
        listing_price = round(rng.uniform(price_low, price_high), 2)
        
        # Seller's cost (what they paid) - typically 40-70% of listing
        item_cost = round(listing_price * rng.uniform(0.40, 0.70), 2)
        
        # Seller's floor - their minimum acceptable (cost + small profit)
        seller_floor = round(item_cost * rng.uniform(1.10, 1.25), 2)
        
        # Buyer's target - based on category discount patterns
        avg_discount = cat_data["avg_discount"]
        buyer_target = round(listing_price * (1 - avg_discount * rng.uniform(0.8, 1.2)), 2)
        
        # Shipping
        shipping = round(rng.uniform(0, 30) if listing_price < 500 else rng.uniform(0, 50), 2)
        
        # Condition
        conditions = ["New", "Like New", "Very Good", "Good", "Acceptable"]
        condition = rng.choice(conditions)
        
        # Description
        descriptions = [
            f"{condition} condition. Ships fast. No returns.",
            f"Authentic {item_name}. {condition}. Message with questions.",
            f"Great {item_name}! {condition} condition. Fast shipping.",
            f"{condition}. Works perfectly. Smoke-free home.",
            f"Selling my {item_name}. {condition}. Price negotiable.",
        ]
        
        scenario = EbayScenario(
            uid=f"ebay_{i:05d}",
            title=item_name,
            category=category,
            condition=condition,
            listing_price=listing_price,
            buyer_target=buyer_target,
            seller_floor=seller_floor,
            item_cost=item_cost,
            shipping=shipping,
            description=rng.choice(descriptions),
            negotiation_style=cat_data["negotiation_style"],
        )
        scenarios.append(scenario)
    
    return scenarios


def load_huggingface_negotiations() -> list[dict]:
    """Load negotiation data from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        
        datasets_to_load = [
            "stanfordnlp/craigslist_bargains",
            "ChicagoHAI/language-of-bargaining",
        ]
        
        all_data = []
        for ds_name in datasets_to_load:
            try:
                ds = load_dataset(ds_name, split="train")
                all_data.extend(list(ds))
                print(f"Loaded {len(ds)} examples from {ds_name}")
            except Exception as e:
                print(f"Could not load {ds_name}: {e}")
        
        return all_data
    except ImportError:
        print("Install datasets: pip install datasets")
        return []


def create_ebay_training_data(n_scenarios: int = 2000, output_file: str = "ebay_training.json"):
    """Create eBay-specific training data file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Generating {n_scenarios} eBay scenarios...")
    scenarios = generate_ebay_scenarios(n_scenarios)
    
    # Convert to training format
    training_data = []
    for sc in scenarios:
        training_data.append({
            "uid": sc.uid,
            "title": sc.title,
            "category": sc.category,
            "condition": sc.condition,
            "listing_price": sc.listing_price,
            "buyer_target": sc.buyer_target,
            "seller_floor": sc.seller_floor,
            "shipping": sc.shipping,
            "description": sc.description,
            "platform": "ebay",
        })
    
    output_path = os.path.join(DATA_DIR, output_file)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Created {output_path} with {len(training_data)} scenarios")
    
    # Print stats
    print("\n📊 Dataset Statistics:")
    for cat in EBAY_CATEGORIES:
        cat_count = sum(1 for d in training_data if d["category"] == cat)
        print(f"   {cat}: {cat_count} scenarios")
    
    return training_data


if __name__ == "__main__":
    print("🛒 eBay Negotiation Data Generator")
    print("=" * 50)
    
    # Generate training data
    data = create_ebay_training_data(2000)
    
    # Show examples
    print("\n📋 Sample Scenarios:")
    for i, sc in enumerate(data[:3]):
        print(f"\n{i+1}. {sc['title']}")
        print(f"   Category: {sc['category']}")
        print(f"   Listing: ${sc['listing_price']:.2f}")
        print(f"   Target: ${sc['buyer_target']:.2f}")
        print(f"   Seller Floor: ${sc['seller_floor']:.2f}")
