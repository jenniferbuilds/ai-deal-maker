"""Multi-platform negotiation framework.

Extends the Craigslist buyer agent to support:
1. E-commerce (eBay, FB Marketplace, Poshmark)
2. Salary/job offer negotiation
3. Rent/housing negotiation
4. Service contract negotiation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from enum import Enum


class NegotiationType(Enum):
    PRICE_ONLY = "price_only"  # Single item price (Craigslist)
    MULTI_ISSUE = "multi_issue"  # Multiple items/terms (Deal or No Deal)
    SALARY = "salary"  # Job offers
    RENT = "rent"  # Housing
    CONTRACT = "contract"  # Service agreements


@dataclass(frozen=True)
class NegotiationScenario:
    """Base class for all negotiation scenarios."""
    uid: str
    negotiation_type: NegotiationType
    description: str
    your_target: float  # What you want
    their_floor: float  # Their secret minimum (unknown to you)
    currency: str = "USD"
    
    # For multi-issue negotiations
    items: dict[str, tuple[int, float]] | None = None  # item_name -> (quantity, your_value)
    
    # Context
    context: str = ""  # Additional info (job details, item condition, etc.)


@dataclass
class Offer:
    """An offer made during negotiation."""
    terms: dict[str, float | str]  # e.g., {"price": 150.0} or {"salary": 120000, "equity": 0.5}
    is_final: bool = False
    expires_in_turns: int | None = None


class PlatformAdapter(Protocol):
    """Adapter for different negotiation platforms."""
    
    def parse_offer(self, text: str) -> Offer | None:
        """Extract structured offer from natural language."""
        ...
    
    def format_offer(self, offer: Offer) -> str:
        """Convert structured offer to natural language."""
        ...
    
    def validate_offer(self, offer: Offer, scenario: NegotiationScenario) -> bool:
        """Check if offer is valid for this scenario."""
        ...
    
    def calculate_reward(self, final_offer: Offer | None, scenario: NegotiationScenario) -> float:
        """Calculate reward for the outcome."""
        ...


# Platform-specific implementations

class EcommerceAdapter:
    """Adapter for e-commerce platforms (eBay, FB Marketplace, etc.)."""
    
    def __init__(self, platform_name: str):
        self.platform = platform_name
    
    def parse_offer(self, text: str) -> Offer | None:
        import re
        # Match price patterns like "$150" or "150 dollars" or "offer 150"
        pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        matches = re.findall(pattern, text)
        if matches:
            price = float(matches[0].replace(',', ''))
            return Offer(terms={"price": price})
        return None
    
    def format_offer(self, offer: Offer) -> str:
        price = offer.terms.get("price", 0)
        if offer.is_final:
            return f"${price:.2f} is my final offer. Take it or leave it."
        return f"How about ${price:.2f}?"
    
    def calculate_reward(self, final_offer: Offer | None, scenario: NegotiationScenario) -> float:
        if final_offer is None:
            return -0.3  # No deal penalty
        
        price = final_offer.terms.get("price", float('inf'))
        target = scenario.your_target
        floor = scenario.their_floor
        
        # Reward: 1.0 = got it for floor price, 0.0 = paid target price
        if price > target:
            return -1.0  # Overpaid
        
        span = max(target - floor, 1e-6)
        return max(-1.0, min(1.0, (target - price) / span))


class SalaryAdapter:
    """Adapter for salary negotiation."""
    
    def parse_offer(self, text: str) -> Offer | None:
        import re
        # Match salary patterns like "$120k" or "120000" or "120,000"
        patterns = [
            r'\$?(\d+(?:,\d{3})*)\s*k?\b',  # $120k or 120000
            r'\$?(\d+)k\b',  # 120k
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).replace(',', '')
                salary = float(val)
                if 'k' in text.lower():
                    salary *= 1000
                return Offer(terms={"base_salary": salary})
        return None
    
    def format_offer(self, offer: Offer) -> str:
        salary = offer.terms.get("base_salary", 0)
        equity = offer.terms.get("equity_percent", 0)
        
        msg = f"I'm looking for a base salary of ${salary:,.0f}"
        if equity > 0:
            msg += f" and {equity}% equity"
        
        if offer.is_final:
            msg += ". This is my final expectation."
        
        return msg + "."
    
    def calculate_reward(self, final_offer: Offer | None, scenario: NegotiationScenario) -> float:
        if final_offer is None:
            return -0.5  # No deal is costly in job search
        
        salary = final_offer.terms.get("base_salary", 0)
        target = scenario.your_target
        floor = scenario.their_floor
        
        # Reward: higher salary = higher reward
        if salary < floor:
            return -0.2  # Accepted too low
        
        span = max(target - floor, 1e-6)
        return max(-1.0, min(1.0, (salary - floor) / span))


class RentAdapter:
    """Adapter for rent/housing negotiation."""
    
    def parse_offer(self, text: str) -> Offer | None:
        import re
        # Match rent patterns
        pattern = r'\$?(\d+(?:,\d{3})*)\s*(?:per\s+month|/mo|monthly)?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            rent = float(match.group(1).replace(',', ''))
            terms = {"monthly_rent": rent}
            
            # Check for lease terms
            lease_match = re.search(r'(\d+)\s*month', text, re.IGNORECASE)
            if lease_match:
                terms["lease_months"] = int(lease_match.group(1))
            
            return Offer(terms=terms)
        return None
    
    def format_offer(self, offer: Offer) -> str:
        rent = offer.terms.get("monthly_rent", 0)
        lease = offer.terms.get("lease_months", 12)
        
        msg = f"I can pay ${rent:,.0f}/month"
        if lease != 12:
            msg += f" for a {lease}-month lease"
        
        return msg + "."
    
    def calculate_reward(self, final_offer: Offer | None, scenario: NegotiationScenario) -> float:
        if final_offer is None:
            return -0.4
        
        rent = final_offer.terms.get("monthly_rent", float('inf'))
        target = scenario.your_target
        floor = scenario.their_floor
        
        span = max(target - floor, 1e-6)
        return max(-1.0, min(1.0, (target - rent) / span))


# Example scenarios

ECOMMERCE_SCENARIO = NegotiationScenario(
    uid="ebay_001",
    negotiation_type=NegotiationType.PRICE_ONLY,
    description="Vintage camera on eBay",
    your_target=150.0,  # You want to pay $150 or less
    their_floor=120.0,  # Seller's minimum (unknown to you)
    currency="USD",
    context="Canon AE-1, excellent condition, includes 50mm lens"
)

SALARY_SCENARIO = NegotiationScenario(
    uid="job_001",
    negotiation_type=NegotiationType.SALARY,
    description="Senior Software Engineer position at TechCorp",
    your_target=180000.0,  # Your target salary
    their_floor=140000.0,  # Company's budget floor
    currency="USD",
    context="5 years experience, distributed systems, remote-friendly"
)

RENT_SCENARIO = NegotiationScenario(
    uid="rent_001",
    negotiation_type=NegotiationType.RENT,
    description="2BR apartment in downtown",
    your_target=2500.0,  # Your max rent
    their_floor=2200.0,  # Landlord's minimum
    currency="USD",
    context="Modern building, gym, parking included"
)


def get_adapter(negotiation_type: NegotiationType) -> PlatformAdapter:
    """Get the appropriate adapter for a negotiation type."""
    adapters = {
        NegotiationType.PRICE_ONLY: EcommerceAdapter("marketplace"),
        NegotiationType.SALARY: SalaryAdapter(),
        NegotiationType.RENT: RentAdapter(),
    }
    return adapters.get(negotiation_type, EcommerceAdapter("generic"))


if __name__ == "__main__":
    # Demo: parse offers from different platforms
    ecommerce = EcommerceAdapter("eBay")
    salary = SalaryAdapter()
    rent = RentAdapter()
    
    print("=== E-commerce Example ===")
    offer = ecommerce.parse_offer("I'll give you $145 for the camera")
    print(f"Parsed: {offer}")
    print(f"Formatted: {ecommerce.format_offer(offer)}")
    
    print("\n=== Salary Example ===")
    offer = salary.parse_offer("We can offer $160k base")
    print(f"Parsed: {offer}")
    print(f"Formatted: {salary.format_offer(offer)}")
    
    print("\n=== Rent Example ===")
    offer = rent.parse_offer("How about $2300/month for 18 months?")
    print(f"Parsed: {offer}")
    print(f"Formatted: {rent.format_offer(offer)}")
