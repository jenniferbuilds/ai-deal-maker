"""Memory integration for negotiation agent using Memorable.

This module integrates procedural memory so the agent can:
1. Remember successful negotiation strategies
2. Learn from past mistakes
3. Adapt tactics based on item category/price range
4. Share knowledge across negotiation sessions

Installation:
    pip install memorable-kg
    memorable login
    memorable enable
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

try:
    from memorable import Memory, Procedure
    HAS_MEMORABLE = True
except ImportError:
    HAS_MEMORABLE = False
    print("Warning: memorable-kg not installed. Memory features disabled.")
    print("Install with: pip install memorable-kg")


@dataclass
class NegotiationMemory:
    """Memory system for negotiation agent."""
    
    memory_store: Optional[object] = None
    
    def __init__(self):
        if HAS_MEMORABLE:
            try:
                self.memory_store = Memory()
            except Exception as e:
                print(f"Could not initialize memory: {e}")
    
    def remember_strategy(
        self,
        scenario_type: str,
        price_range: tuple[float, float],
        successful_tactics: list[str],
        final_price_ratio: float,
        outcome: str
    ) -> None:
        """Store a successful negotiation strategy."""
        if not self.memory_store:
            return
        
        procedure = Procedure(
            name=f"negotiate_{scenario_type}_{price_range[0]}_{price_range[1]}",
            description=f"Strategy for {scenario_type} items priced ${price_range[0]}-${price_range[1]}",
            steps=successful_tactics,
            metadata={
                "scenario_type": scenario_type,
                "price_range": price_range,
                "final_price_ratio": final_price_ratio,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            self.memory_store.remember(procedure)
        except Exception as e:
            print(f"Failed to store memory: {e}")
    
    def recall_strategy(self, scenario_type: str, price: float) -> Optional[dict]:
        """Recall similar negotiation strategies."""
        if not self.memory_store:
            return None
        
        try:
            # Search for similar scenarios
            query = f"negotiate {scenario_type} around ${price}"
            procedures = self.memory_store.recall(query, k=3)
            
            if procedures:
                # Return the most relevant strategy
                best = procedures[0]
                return {
                    "tactics": best.steps,
                    "expected_ratio": best.metadata.get("final_price_ratio", 0.8),
                    "past_outcomes": [p.metadata.get("outcome") for p in procedures]
                }
        except Exception as e:
            print(f"Failed to recall memory: {e}")
        
        return None
    
    def learn_from_episode(self, episode: dict) -> None:
        """Extract and store lessons from a negotiation episode."""
        if not episode.get("success"):
            # Store failure pattern to avoid
            self.remember_failure(
                scenario_type=episode.get("category", "unknown"),
                price_range=(episode.get("listing_price", 0) * 0.8, episode.get("listing_price", 0) * 1.2),
                mistakes=episode.get("mistakes", []),
                outcome=episode.get("end_reason", "unknown")
            )
        else:
            # Store successful strategy
            self.remember_strategy(
                scenario_type=episode.get("category", "unknown"),
                price_range=(episode.get("listing_price", 0) * 0.8, episode.get("listing_price", 0) * 1.2),
                successful_tactics=episode.get("tactics", []),
                final_price_ratio=episode.get("final_price", 0) / episode.get("listing_price", 1),
                outcome="deal"
            )
    
    def remember_failure(
        self,
        scenario_type: str,
        price_range: tuple[float, float],
        mistakes: list[str],
        outcome: str
    ) -> None:
        """Store patterns that led to failure."""
        if not self.memory_store:
            return
        
        procedure = Procedure(
            name=f"avoid_{scenario_type}_{outcome}",
            description=f"Patterns to avoid for {scenario_type} items",
            steps=mistakes,
            metadata={
                "scenario_type": scenario_type,
                "price_range": price_range,
                "outcome": outcome,
                "type": "failure_pattern",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            self.memory_store.remember(procedure)
        except Exception as e:
            print(f"Failed to store failure memory: {e}")


class MemoryEnhancedBuyer:
    """Buyer agent with procedural memory."""
    
    def __init__(self, base_buyer, memory: Optional[NegotiationMemory] = None):
        self.base_buyer = base_buyer
        self.memory = memory or NegotiationMemory()
    
    def act(self, conversations: list[list[dict]]) -> list:
        """Generate responses with memory-informed context."""
        # Try to recall relevant strategies
        enhanced_conversations = []
        
        for msgs in conversations:
            # Extract scenario info from system prompt
            system_msg = msgs[0]["content"] if msgs else ""
            scenario_type = self._extract_category(system_msg)
            price = self._extract_price(system_msg)
            
            # Recall similar negotiations
            past_strategy = self.memory.recall_strategy(scenario_type, price)
            
            if past_strategy:
                # Inject memory hint into conversation
                hint = self._format_memory_hint(past_strategy)
                enhanced_msgs = [{"role": "system", "content": system_msg + f"\n\nMemory hint: {hint}"}] + msgs[1:]
                enhanced_conversations.append(enhanced_msgs)
            else:
                enhanced_conversations.append(msgs)
        
        # Get base buyer's actions
        generations = self.base_buyer.act(enhanced_conversations)
        
        return generations
    
    def _extract_category(self, system_msg: str) -> str:
        """Extract item category from system prompt."""
        if "Category:" in system_msg:
            return system_msg.split("Category:")[1].split("\n")[0].strip()
        return "unknown"
    
    def _extract_price(self, system_msg: str) -> float:
        """Extract listing price from system prompt."""
        import re
        match = re.search(r"Listing price: \$?([\d,]+)", system_msg)
        if match:
            return float(match.group(1).replace(",", ""))
        return 0.0
    
    def _format_memory_hint(self, strategy: dict) -> str:
        """Format recalled strategy as a hint."""
        tactics = strategy.get("tactics", [])
        ratio = strategy.get("expected_ratio", 0.8)
        
        hint = f"In similar negotiations, successful tactics included: {', '.join(tactics[:3])}. "
        hint += f"Typical deal price was around {ratio*100:.0f}% of listing."
        
        return hint


# Standalone memory functions for when memorable isn't installed

class SimpleMemory:
    """Simple file-based memory fallback."""
    
    def __init__(self, memory_file: str = "negotiation_memory.json"):
        self.memory_file = memory_file
        self.memories = self._load_memories()
    
    def _load_memories(self) -> list[dict]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file) as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_memories(self) -> None:
        with open(self.memory_file, "w") as f:
            json.dump(self.memories, f, indent=2)
    
    def remember(self, scenario: dict) -> None:
        """Store a negotiation outcome."""
        self.memories.append({
            **scenario,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 1000 memories
        self.memories = self.memories[-1000:]
        self._save_memories()
    
    def recall_similar(self, category: str, price_range: tuple[float, float], k: int = 5) -> list[dict]:
        """Find similar past negotiations."""
        similar = []
        for mem in reversed(self.memories):
            if mem.get("category") == category:
                mem_price = mem.get("listing_price", 0)
                if price_range[0] <= mem_price <= price_range[1]:
                    similar.append(mem)
                    if len(similar) >= k:
                        break
        return similar


if __name__ == "__main__":
    # Demo: using memory-enhanced negotiation
    print("=== Memory-Enhanced Negotiation Agent ===\n")
    
    if HAS_MEMORABLE:
        print("✓ Memorable integration available")
        memory = NegotiationMemory()
    else:
        print("✗ Using simple file-based memory fallback")
        memory = SimpleMemory()
    
    # Example: store a successful negotiation
    example_episode = {
        "category": "electronics",
        "listing_price": 200,
        "final_price": 160,
        "tactics": ["started low at $120", "incremented by $20", "mentioned competitor pricing"],
        "success": True
    }
    
    print("\nStoring example negotiation:")
    print(f"  Category: {example_episode['category']}")
    print(f"  Listing: ${example_episode['listing_price']}")
    print(f"  Final: ${example_episode['final_price']}")
    print(f"  Tactics: {example_episode['tactics']}")
    
    if isinstance(memory, SimpleMemory):
        memory.remember(example_episode)
        print("\n✓ Stored in simple memory")
        
        # Recall similar
        similar = memory.recall_similar("electronics", (150, 250))
        print(f"\nRecalled {len(similar)} similar negotiations")
    else:
        memory.learn_from_episode(example_episode)
        print("\n✓ Stored in Memorable")
    
    print("\n=== Integration with River Buyer ===")
    print("To use memory-enhanced buyer:")
    print("  1. Create base buyer: buyer = RiverBuyer(...)")
    print("  2. Wrap with memory: memory_buyer = MemoryEnhancedBuyer(buyer, memory)")
    print("  3. Use normally: episodes = run_episodes(scenarios, memory_buyer, voice)")
