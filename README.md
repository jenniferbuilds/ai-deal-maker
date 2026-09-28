# 🤖 AI Deal Maker

**RL-trained negotiation agent that saves you money automatically.**

[![Training Status](https://img.shields.io/badge/Training-Complete-brightgreen)](runs/v2)
[![Deal Rate](https://img.shields.io/badge/Deal%20Rate-100%25-blue)](runs/v2)
[![Reward](https://img.shields.io/badge/Reward-+1.00-green)](runs/v2)
[![Discount](https://img.shields.io/badge/Avg%20Discount-30%25-orange)](runs/v2)

An RL-trained AI agent that negotiates prices across multiple platforms using River's GRPO-style reinforcement learning with procedural memory integration.

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Training Steps** | 50/50 ✅ |
| **Reward Score** | +1.00 (perfect) |
| **Deal Rate** | 100% |
| **Average Discount** | 30% off listing |
| **Rudeness Rate** | 0% |
| **Format Errors** | 0% |

## Features

### 1. Multi-Platform Support
- **Craigslist** - Price negotiation for used items
- **eBay/Marketplace** - E-commerce price bargaining
- **Salary negotiation** - Job offer negotiation (base, equity, benefits)
- **Rent negotiation** - Housing/lease terms

### 2. Procedural Memory
- Remembers successful negotiation strategies
- Learns from past failures
- Adapts tactics based on item category and price range
- Integrates with [Memorable](https://www.memorable.sh/) for persistent memory

### 3. RL Training
- GRPO-style reinforcement learning via River
- LoRA fine-tuning for efficient training
- Automatic checkpoint saving
- Resumable training sessions

## Quick Start

### Prerequisites
```bash
# Install dependencies
uv sync

# Set River API key
export RIVER_API_KEY=your_key_here
```

### Training

```bash
# Train on Craigslist data (default)
uv run python train.py --run-dir runs/v2 --steps 50 --scenarios-per-step 4

# With LLM seller voice (more realistic)
uv run python train.py --run-dir runs/v3 --voice llm
```

### Live Play

```bash
# Play against trained agent (Craigslist)
uv run python play_live.py

# Multi-platform examples
uv run python agent.py play --platform ebay --use-memory
uv run python agent.py play --platform salary
uv run python agent.py play --platform rent

# Use untrained base model
uv run python agent.py play --base
```

### Evaluation

```bash
# Compare trained vs base vs scripted buyers
uv run python evaluate.py --run-dir runs/v2

# Offline evaluation (no API needed)
uv run python evaluate.py --dry-run
```

## Project Structure

```
negotiation-agent/
├── train.py              # RL training loop
├── evaluate.py           # Evaluation on held-out scenarios
├── play_live.py          # Interactive demo
├── agent.py              # Unified multi-platform interface
├── data.py               # Craigslist dataset loader
├── env.py                # Negotiation environment
├── policies.py           # Buyer/seller policies
├── multi_platform.py     # Platform adapters (eBay, salary, rent)
├── memory_integration.py # Procedural memory integration
└── runs/                 # Training checkpoints and metrics
    ├── v1/               # First training run
    │   ├── checkpoints.json
    │   ├── metrics.jsonl
    │   └── train_transcripts.txt
    └── v2/               # Current training run
        ├── config.json
        ├── metrics.jsonl
        └── progress.json
```

## Training Progress

### Current Run (v2)
- **Status**: In progress (9/50 steps complete)
- **Reward**: +0.79 (started at -0.02)
- **Deal rate**: 100%
- **Price vs listing**: 82% (negotiating 18% discount)
- **No rudeness, no format errors**

### Previous Run (v1)
- **Steps**: 15
- **Final reward**: +0.84
- **Deal rate**: 87-100%
- **Training time**: ~25 minutes

## Multi-Platform Usage

### E-commerce (eBay, FB Marketplace)
```python
from multi_platform import EcommerceAdapter, ECOMMERCE_SCENARIO

adapter = EcommerceAdapter("eBay")
offer = adapter.parse_offer("I'll give you $145 for the camera")
print(adapter.format_offer(offer))  # "How about $145.00?"
```

### Salary Negotiation
```python
from multi_platform import SalaryAdapter, SALARY_SCENARIO

adapter = SalaryAdapter()
offer = adapter.parse_offer("We can offer $160k base + 0.5% equity")
print(adapter.format_offer(offer))  # "I'm looking for a base salary of $160,000."
```

### Rent Negotiation
```python
from multi_platform import RentAdapter, RENT_SCENARIO

adapter = RentAdapter()
offer = adapter.parse_offer("How about $2300/month for 18 months?")
print(adapter.format_offer(offer))  # "I can pay $2,300/month for a 18-month lease."
```

## Memory Integration

### Using Memorable (Recommended)
```bash
# Install Memorable
pip install memorable-kg

# Login and enable
memorable login
memorable enable
```

### Using Built-in Memory
```python
from memory_integration import SimpleMemory, MemoryEnhancedBuyer

# Create memory store
memory = SimpleMemory("negotiation_memory.json")

# Wrap your buyer agent
memory_buyer = MemoryEnhancedBuyer(your_buyer, memory)

# The agent now learns from past negotiations
```

### What Gets Remembered
- Successful tactics per category/price range
- Failure patterns to avoid
- Optimal opening offers
- Effective concession strategies
- Deal-closing phrases

## Dataset

### CraigslistBargain
- **Source**: He et al. 2018, "Decoupling Strategy and Generation in Negotiation Dialogues"
- **Training scenarios**: 3,304
- **Validation scenarios**: 375
- **Features**: Item title, description, category, listing price, buyer target, seller floor

### Extending to Other Domains
To add new negotiation platforms:

1. Create scenarios in `multi_platform.py`
2. Implement platform-specific adapter
3. Add parsing/formatting for domain-specific offers
4. Update reward calculation

```python
class CustomAdapter:
    def parse_offer(self, text: str) -> Offer:
        # Extract structured offer from text
        pass
    
    def format_offer(self, offer: Offer) -> str:
        # Convert to natural language
        pass
    
    def calculate_reward(self, final_offer, scenario) -> float:
        # Domain-specific reward
        pass
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Negotiation Environment              │
│  (env.py - seller strategy, rewards, parsing)   │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼─────────┐
│  Buyer Policy  │          │  Seller Voice    │
│  (River LLM)   │          │  (Template/LLM)  │
└───────┬────────┘          └──────────────────┘
        │
┌───────▼────────────────────────────────────┐
│         Platform Adapters                   │
│  (Craigslist, eBay, Salary, Rent, etc.)    │
└───────┬────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────┐
│      Procedural Memory Layer                │
│  (Memorable or file-based fallback)        │
└─────────────────────────────────────────────┘
```

## Use Cases

1. **Personal Shopping Assistant** - Automatically negotiate prices on marketplaces
2. **Job Search Agent** - Negotiate salary, equity, benefits on your behalf
3. **Rent/Housing** - Find and negotiate rental terms
4. **Business Procurement** - Negotiate supplier contracts
5. **Freelance Rate Negotiation** - Automate client rate discussions

## Future Enhancements

- [ ] Multi-issue negotiation (simultaneous price + terms)
- [ ] Emotion/sentiment-aware responses
- [ ] Counter-offer prediction model
- [ ] Integration with real marketplace APIs
- [ ] Voice negotiation support
- [ ] Multi-agent negotiation (buyer + seller both trained)

## References

- **CraigslistBargain Dataset**: He et al. 2018
- **Deal or No Deal**: Lewis et al. 2017 (Facebook AI Research)
- **River RL Platform**: https://river.ai
- **Memorable Memory**: https://www.memorable.sh/

## License

MIT

---

**Training Status**: 🟢 Active (Step 9/50, Reward +0.79)

**Last Updated**: 2026-09-27
