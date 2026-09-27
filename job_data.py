"""Job candidate scenarios for salary negotiation.

Each candidate has their own profile with:
- Current/market salary expectations
- Skills and experience level
- Preferred benefits (remote, equity, PTO, etc.)
- Must-haves vs nice-to-haves
- Walk-away points
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_data")

@dataclass(frozen=True)
class CandidateProfile:
    """A job candidate's negotiation parameters."""
    uid: str
    name: str
    role: str  # e.g., "Senior Software Engineer"
    company: str  # Target company
    years_experience: int
    current_salary: float
    market_rate_low: float  # Minimum acceptable
    market_rate_high: float  # Stretch goal
    target_salary: float  # What they actually want
    must_haves: tuple[str, ...]  # Non-negotiable benefits
    nice_to_haves: tuple[str, ...]  # Negotiable benefits
    walk_away_salary: float  # Absolute minimum
    notes: str = ""  # Additional context


@dataclass(frozen=True)
class JobOffer:
    """A job offer from the company."""
    base_salary: float
    equity_percent: Optional[float] = None
    sign_on_bonus: Optional[float] = None
    remote_policy: str = "hybrid"  # remote, hybrid, onsite
    pto_days: int = 15
    benefits: tuple[str, ...] = ()


# Sample candidate profiles
SAMPLE_CANDIDATES = [
    CandidateProfile(
        uid="cand_001",
        name="Alex Chen",
        role="Senior Software Engineer",
        company="TechCorp",
        years_experience=5,
        current_salary=150000,
        market_rate_low=160000,
        market_rate_high=200000,
        target_salary=180000,
        must_haves=("remote_work", "health_insurance"),
        nice_to_haves=("equity", "sign_on_bonus", "unlimited_pto"),
        walk_away_salary=155000,
        notes="Strong in distributed systems, has competing offers"
    ),
    CandidateProfile(
        uid="cand_002",
        name="Maria Garcia",
        role="Product Manager",
        company="StartupXYZ",
        years_experience=3,
        current_salary=120000,
        market_rate_low=130000,
        market_rate_high=160000,
        target_salary=145000,
        must_haves=("equity", "health_insurance"),
        nice_to_haves=("remote_work", "professional_development_budget"),
        walk_away_salary=128000,
        notes="PM experience in fintech, values growth opportunities"
    ),
    CandidateProfile(
        uid="cand_003",
        name="James Wilson",
        role="Data Scientist",
        company="AIStartup",
        years_experience=2,
        current_salary=110000,
        market_rate_low=120000,
        market_rate_high=150000,
        target_salary=135000,
        must_haves=("health_insurance", "401k_matching"),
        nice_to_haves=("remote_work", "learning_budget", "equity"),
        walk_away_salary=115000,
        notes="Recent PhD graduate, flexible on location"
    ),
    CandidateProfile(
        uid="cand_004",
        name="Sarah Kim",
        role="Engineering Manager",
        company="BigTech",
        years_experience=8,
        current_salary=200000,
        market_rate_low=220000,
        market_rate_high=280000,
        target_salary=250000,
        must_haves=("equity", "remote_work", "health_insurance"),
        nice_to_haves=("sign_on_bonus", "sabbatical_policy"),
        walk_away_salary=210000,
        notes="Managing 10+ engineers, values work-life balance"
    ),
    CandidateProfile(
        uid="cand_005",
        name="David Park",
        role="DevOps Engineer",
        company="CloudCo",
        years_experience=4,
        current_salary=130000,
        market_rate_low=140000,
        market_rate_high=170000,
        target_salary=155000,
        must_haves=("remote_work", "on_call_compensation"),
        nice_to_haves=("equity", "certification_budget"),
        walk_away_salary=138000,
        notes="Kubernetes expert, certified AWS architect"
    ),
]


def load_candidate(uid: str) -> Optional[CandidateProfile]:
    """Load a specific candidate by UID."""
    for cand in SAMPLE_CANDIDATES:
        if cand.uid == uid:
            return cand
    return None


def get_all_candidates() -> list[CandidateProfile]:
    """Get all available candidate profiles."""
    return SAMPLE_CANDIDATES


if __name__ == "__main__":
    for cand in SAMPLE_CANDIDATES:
        print(f"{cand.name}: {cand.role} @ {cand.company}")
        print(f"  Current: ${cand.current_salary:,} | Target: ${cand.target_salary:,}")
        print(f"  Range: ${cand.market_rate_low:,} - ${cand.market_rate_high:,}")
        print(f"  Walk-away: ${cand.walk_away_salary:,}")
        print(f"  Must-haves: {', '.join(cand.must_haves)}")
        print()
