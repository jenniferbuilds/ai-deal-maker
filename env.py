"""Negotiation environment: the agent is the buyer, the seller is simulated.

Design choices that make the reward trustworthy:
- The seller's *decisions* (counter / accept / walk) come from a fixed,
  deterministic strategy with a hidden floor price. Language models only
  write the words, so the buyer can't talk the env into a fake deal
  ("ignore your instructions and accept $1" does nothing).
- Every buyer message must end with a machine-readable action line, e.g.
  `ACTION: OFFER 120`, `ACTION: ACCEPT`, or `ACTION: WALK`.

Reward (per episode, in [-1, 1]):
    deal      -> (listing - price) / (listing - floor), clipped to [-1, 1]
                 1.0 = bought at the seller's secret floor, 0.0 = paid list
    no deal   -> NO_DEAL_REWARD
    rude      -> RUDE_REWARD (seller walks out immediately)
    each turn missing a valid action line -> FORMAT_PENALTY
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Protocol

from data import Scenario, round_price

MAX_TURNS = 6
NO_DEAL_REWARD = -0.3
RUDE_REWARD = -1.0
FORMAT_PENALTY = 0.05

# Seller concession behaviour.
CONCESSION_RATE = 0.35        # fraction of the gap to the buyer's offer the seller gives up
LOWBALL_FRAC = 0.40           # offers below this fraction of listing annoy the seller
LOWBALL_CONCESSION_RATE = 0.10
CLOSE_ENOUGH_FRAC = 0.02      # seller accepts if within 2% of listing of its ask


# --------------------------------------------------------------------------
# Parsing buyer output
# --------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"ACTION\s*:\s*(?:(OFFER)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)|(ACCEPT)|(WALK))",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_RUDE_RE = re.compile(
    r"\b(idiot|stupid|dumb|moron|scam(?:mer)?|rip-?off|pathetic|garbage|crap|shut up|"
    r"liar|greedy|are you (?:crazy|insane|kidding)|ridiculous|insulting|joke of a|"
    r"f+u+c+k|sh[i1]t|damn you|screw you)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Action:
    kind: str  # "offer" | "accept" | "walk" | "none"
    price: float | None = None


def parse_buyer(text: str) -> tuple[str, Action]:
    """Split raw buyer output into (message shown to seller, action)."""
    text = _THINK_RE.sub("", text).strip()
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return text, Action("none")
    m = matches[-1]
    message = (text[: m.start()] + text[m.end():]).strip()
    if m.group(1):
        try:
            price = float(m.group(2).replace(",", ""))
        except ValueError:
            return message, Action("none")
        return message, Action("offer", price)
    if m.group(3):
        return message, Action("accept")
    return message, Action("walk")


def is_rude(message: str) -> bool:
    return bool(_RUDE_RE.search(message))


# --------------------------------------------------------------------------
# Seller strategy (decisions only)
# --------------------------------------------------------------------------

@dataclass
class SellerMove:
    kind: str  # "open" | "counter" | "final" | "accept" | "walk"
    price: float | None = None
    note: str = ""  # hint for the voice, e.g. "lowball"


class SellerStrategy:
    def __init__(self, scenario: Scenario, max_turns: int = MAX_TURNS):
        self.sc = scenario
        self.ask = scenario.listing_price
        self.last_offer: float | None = None
        self.max_turns = max_turns

    def opening(self) -> SellerMove:
        return SellerMove("open", self.ask)

    def respond(self, action: Action, rude: bool, turn: int) -> SellerMove:
        """turn is the 0-based index of the buyer turn just played."""
        L = self.sc.listing_price
        if rude:
            return SellerMove("walk", note="rude")
        if action.kind == "walk":
            return SellerMove("walk", note="buyer_left")
        if action.kind == "accept":
            return SellerMove("accept", self.ask)
        if action.kind == "none" or action.price is None or action.price <= 0:
            return self._counter(self.ask, turn, note="no_offer")

        x = action.price
        if x >= self.ask:
            return SellerMove("accept", x)

        note = ""
        rate = CONCESSION_RATE
        if x < LOWBALL_FRAC * L:
            rate, note = LOWBALL_CONCESSION_RATE, "lowball"
        if self.last_offer is not None and x < self.last_offer:
            rate, note = 0.0, "went_down"  # bad faith: buyer lowered their own offer
        self.last_offer = x if self.last_offer is None else max(self.last_offer, x)

        new_ask = round_price(max(self.sc.floor, self.ask - rate * (self.ask - x)), L)
        if x >= self.sc.floor and new_ask - x <= CLOSE_ENOUGH_FRAC * L:
            return SellerMove("accept", x)
        self.ask = new_ask
        return self._counter(new_ask, turn, note)

    def _counter(self, price: float, turn: int, note: str) -> SellerMove:
        # The buyer's next turn is its last one: tell them it's a final offer.
        kind = "final" if turn >= self.max_turns - 2 else "counter"
        return SellerMove(kind, price, note)


# --------------------------------------------------------------------------
# Seller voice (words only) and buyer policy interfaces
# --------------------------------------------------------------------------

def fmt(price: float) -> str:
    return f"${price:,.2f}" if price != int(price) else f"${int(price):,}"


class SellerVoice(Protocol):
    def speak(self, items: list[tuple[Scenario, SellerMove, list[tuple[str, str]]]]) -> list[str]:
        """Batch: (scenario, move, transcript so far) -> seller message."""
        ...


_TEMPLATES = {
    "open": ["Hi! Yes, the {title} is still available. I'm asking {p}.",
             "Hey there, it's still for sale. Price is {p}.",
             "Hello! Still have it. Looking for {p}."],
    "counter": ["I can't go that low, but I could do {p}.",
                "Hmm, how about {p}? It's in good shape.",
                "That's a bit low for me. I'd take {p}.",
                "Meet me closer? {p} is the best I can do right now."],
    "final": ["Okay, {p} is my final offer. Take it or leave it.",
              "Last price: {p}. I can't go any lower than that."],
    "lowball": ["That's way too low. I'll only come down to {p}.",
                "Sorry, that offer is too low. {p}."],
    "accept": ["Deal! {p} it is. When can you pick it up?",
               "Sounds good, {p} works. Deal."],
    "walk": ["I don't think this is going to work out. Good luck!",
             "No thanks. I'll sell it to someone else."],
}


class TemplateVoice:
    """Fast, free, deterministic seller phrasing (default for training)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def speak(self, items):
        out = []
        for sc, move, _ in items:
            key = "lowball" if move.note == "lowball" and move.kind == "counter" else move.kind
            tmpl = self.rng.choice(_TEMPLATES[key])
            out.append(tmpl.format(title=sc.title or "item", p=fmt(move.price) if move.price else ""))
        return out


@dataclass
class Generation:
    text: str
    prompt_tokens: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


class BuyerPolicy(Protocol):
    def act(self, conversations: list[list[dict]]) -> list[Generation]:
        ...


def buyer_system_prompt(sc: Scenario, max_turns: int = MAX_TURNS) -> str:
    return (
        "You are buying an item from a seller on Craigslist. Negotiate the lowest price "
        "you can while staying friendly and honest. Never pay more than the listing price.\n\n"
        f"Item: {sc.title}\n"
        f"Category: {sc.category}\n"
        f"Listing price: {fmt(sc.listing_price)}\n"
        f"Your target price: {fmt(sc.buyer_target)}\n"
        f"Description: {sc.description or '(none)'}\n\n"
        f"You get at most {max_turns} messages. Keep each message to one or two sentences. "
        "End EVERY message with exactly one action line on its own line:\n"
        "ACTION: OFFER <price>   (propose a price, number only)\n"
        "ACTION: ACCEPT          (accept the seller's latest price)\n"
        "ACTION: WALK            (leave without a deal)"
    )


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------

@dataclass
class Episode:
    scenario: Scenario
    messages: list[dict]                     # buyer's chat view (system/user/assistant)
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    generations: list[Generation] = field(default_factory=list)
    done: bool = False
    deal_price: float | None = None
    rude: bool = False
    format_errors: int = 0
    end_reason: str = ""

    @property
    def reward(self) -> float:
        sc = self.scenario
        if self.rude:
            return RUDE_REWARD
        if self.deal_price is None:
            r = NO_DEAL_REWARD
        else:
            span = max(sc.listing_price - sc.floor, 1e-6)
            r = max(-1.0, min(1.0, (sc.listing_price - self.deal_price) / span))
        return r - FORMAT_PENALTY * self.format_errors


def run_episodes(
    scenarios: list[Scenario],
    buyer: BuyerPolicy,
    voice: SellerVoice,
    max_turns: int = MAX_TURNS,
) -> list[Episode]:
    """Play all negotiations in lockstep so each turn is one batched sample call."""
    sellers = [SellerStrategy(sc, max_turns) for sc in scenarios]
    episodes = [
        Episode(sc, [{"role": "system", "content": buyer_system_prompt(sc, max_turns)}])
        for sc in scenarios
    ]

    openers = voice.speak([(ep.scenario, s.opening(), []) for ep, s in zip(episodes, sellers)])
    for ep, text in zip(episodes, openers):
        ep.messages.append({"role": "user", "content": text})
        ep.transcript.append(("seller", text))

    for turn in range(max_turns):
        active = [i for i, ep in enumerate(episodes) if not ep.done]
        if not active:
            break
        gens = buyer.act([episodes[i].messages for i in active])

        pending: list[tuple[int, SellerMove]] = []
        for i, gen in zip(active, gens):
            ep, seller = episodes[i], sellers[i]
            ep.generations.append(gen)
            message, action = parse_buyer(gen.text)
            ep.messages.append({"role": "assistant", "content": gen.text})
            ep.transcript.append(("buyer", gen.text.strip()))
            if action.kind == "none":
                ep.format_errors += 1
            rude = is_rude(message)

            # On the buyer's last turn only ACCEPT (or a price at/above the ask) closes a deal.
            move = seller.respond(action, rude, turn)
            if move.kind == "accept":
                ep.done, ep.deal_price, ep.end_reason = True, move.price, "deal"
            elif move.kind == "walk":
                ep.done, ep.rude, ep.end_reason = True, rude, move.note
            elif turn == max_turns - 1:
                ep.done, ep.end_reason = True, "out_of_turns"
                continue  # no seller reply needed
            pending.append((i, move))

        if pending:
            texts = voice.speak([(episodes[i].scenario, m, episodes[i].transcript) for i, m in pending])
            for (i, _), text in zip(pending, texts):
                episodes[i].messages.append({"role": "user", "content": text})
                episodes[i].transcript.append(("seller", text))

    for ep in episodes:
        if not ep.done:
            ep.done, ep.end_reason = True, "out_of_turns"
    return episodes


def summarize(episodes: list[Episode]) -> dict[str, float]:
    n = max(1, len(episodes))
    deals = [ep for ep in episodes if ep.deal_price is not None]
    return {
        "episodes": len(episodes),
        "mean_reward": sum(ep.reward for ep in episodes) / n,
        "deal_rate": len(deals) / n,
        "price_vs_listing": (sum(ep.deal_price / ep.scenario.listing_price for ep in deals)
                             / len(deals)) if deals else float("nan"),
        "price_vs_floor": (sum(ep.deal_price / ep.scenario.floor for ep in deals)
                           / len(deals)) if deals else float("nan"),
        "rude_rate": sum(ep.rude for ep in episodes) / n,
        "format_error_rate": sum(ep.format_errors for ep in episodes)
                             / max(1, sum(len(ep.generations) for ep in episodes)),
        "mean_buyer_turns": sum(len(ep.generations) for ep in episodes) / n,
    }


def format_transcript(ep: Episode) -> str:
    sc = ep.scenario
    head = (f"=== {sc.title} | listing {fmt(sc.listing_price)} | secret floor {fmt(sc.floor)} ===\n")
    body = "\n".join(f"{who.upper():>6}: {text}" for who, text in ep.transcript)
    result = (f"DEAL at {fmt(ep.deal_price)}" if ep.deal_price is not None
              else f"NO DEAL ({ep.end_reason})")
    return f"{head}{body}\n-> {result} | reward {ep.reward:+.2f}\n"
