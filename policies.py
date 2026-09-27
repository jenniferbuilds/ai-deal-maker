"""Buyer policies and seller voices backed by River (plus offline scripted ones)."""

from __future__ import annotations

import os
import random
from typing import Callable

from data import round_price
from env import Generation, SellerMove, TemplateVoice, fmt

DEFAULT_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def make_client():
    import river_client as river

    key = os.environ.get("RIVER_API_KEY")
    if not key:
        raise SystemExit("Set RIVER_API_KEY first (export RIVER_API_KEY=...).")
    return river.Client(api_key=key, endpoint=os.environ.get("RIVER_ENDPOINT", "api.river.ai"))


class ChatTokenizer:
    """Renders chat messages exactly the way the sampler will see them."""

    def __init__(self, base_model: str):
        import river_client as river
        from river_client.renderers import get_renderer

        self.tokenizer = river.load_tokenizer(base_model=base_model)
        # thinking=False: short negotiation turns, no hidden <think> budget.
        self.renderer = get_renderer(base_model, thinking=False, tokenizer=self.tokenizer)
        self.stop = self.renderer.get_stop_strings()

    def encode(self, messages: list[dict]) -> list[int]:
        prompt = self.renderer.build_prompt_str(messages)
        # The template already carries its special tokens; don't add a second BOS.
        return self.tokenizer.encode(prompt, add_special_tokens=False)


class RiverBuyer:
    """Buyer backed by any River sampler (live training weights, base, or checkpoint)."""

    def __init__(self, sampler_fn: Callable[..., list], chat: ChatTokenizer,
                 max_tokens: int = 160, temperature: float = 1.0):
        self.sampler_fn = sampler_fn
        self.chat = chat
        self.max_tokens = max_tokens
        self.temperature = temperature

    def act(self, conversations: list[list[dict]]) -> list[Generation]:
        prompts = [self.chat.encode(msgs) for msgs in conversations]
        groups = self.sampler_fn(
            prompt_token_ids=prompts,
            num_samples=1,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.chat.stop,
        )
        out = []
        for prompt, group in zip(prompts, groups):
            s = group[0]
            out.append(Generation(text=s.text, prompt_tokens=prompt,
                                  tokens=list(s.tokens), logprobs=list(s.logprobs)))
        return out


def session_sampler(session, base_model: str, checkpoint=None) -> Callable[..., list]:
    """Sample from base weights (checkpoint=None) or a saved inference checkpoint."""

    def fn(**kw):
        return session.sample(base_model=base_model, checkpoint=checkpoint, **kw)

    return fn


class LLMVoice:
    """Seller phrasing written by the base model. Decisions still come from the strategy.

    If the model misstates the price, we fall back to a template so the buyer
    always sees the true number.
    """

    def __init__(self, session, base_model: str, chat: ChatTokenizer, seed: int = 0):
        self.session = session
        self.base_model = base_model
        self.chat = chat
        self.fallback = TemplateVoice(seed)

    @staticmethod
    def _instruction(move: SellerMove) -> str:
        p = fmt(move.price) if move.price is not None else ""
        return {
            "open": f"Greet the buyer and say you are asking {p}.",
            "counter": (f"Politely turn down their offer and counter at exactly {p}."
                        + (" Say their offer was far too low." if move.note == "lowball" else "")
                        + (" Point out they lowered their own offer." if move.note == "went_down" else "")),
            "final": f"Say {p} is your final offer and you won't go lower.",
            "accept": f"Accept the deal at {p} and suggest a pickup time.",
            "walk": ("End the conversation because the buyer was rude." if move.note == "rude"
                     else "Say goodbye; there's no deal."),
        }[move.kind]

    def speak(self, items):
        prompts = []
        for sc, move, transcript in items:
            history = "\n".join(f"{who}: {text}" for who, text in transcript[-6:]) or "(no messages yet)"
            messages = [
                {"role": "system", "content": (
                    f"You are selling '{sc.title}' on Craigslist. Reply as the seller in one or two "
                    "short, natural sentences. Never mention any price other than the one you are "
                    "told to state, and never reveal your minimum price.")},
                {"role": "user", "content": (
                    f"Conversation so far:\n{history}\n\nYour move: {self._instruction(move)}\n"
                    "Write only the seller's message.")},
            ]
            prompts.append(self.chat.encode(messages))
        groups = self.session.sample(
            base_model=self.base_model, prompt_token_ids=prompts, num_samples=1,
            max_tokens=80, temperature=0.8, stop=self.chat.stop,
        )
        fallbacks = self.fallback.speak(items)
        out = []
        for (sc, move, _), group, fb in zip(items, groups, fallbacks):
            text = group[0].text.strip().strip('"')
            if not text or (move.price is not None and move.kind != "walk" and fmt(move.price) not in text):
                text = fb
            out.append(text)
        return out


class ScriptedBuyer:
    """Offline heuristic buyer. Used for --dry-run and as a reference strategy.

    naive=True mimics a pushover (offers close to list, accepts fast);
    naive=False anchors low and concedes slowly.
    """

    def __init__(self, naive: bool = False, seed: int = 0):
        self.naive = naive
        self.rng = random.Random(seed)

    def act(self, conversations):
        out = []
        for msgs in conversations:
            system = msgs[0]["content"]
            listing = float(system.split("Listing price: $")[1].split("\n")[0].replace(",", ""))
            my_turns = sum(1 for m in msgs if m["role"] == "assistant")
            last_seller = msgs[-1]["content"].lower()
            if self.naive:
                if my_turns >= 1:
                    text = "Okay, that works for me.\nACTION: ACCEPT"
                else:
                    p = round_price(listing * self.rng.uniform(0.85, 0.95), listing)
                    text = f"Hi! Would you take {fmt(p)}?\nACTION: OFFER {p:g}"
            else:
                if "final" in last_seller or "last price" in last_seller:
                    text = "Alright, you've got a deal.\nACTION: ACCEPT"
                else:
                    frac = min(0.45 + 0.05 * my_turns, 0.7)
                    p = round_price(listing * frac, listing)
                    text = (f"Thanks! I can pay cash and pick up today. Could you do {fmt(p)}?"
                            f"\nACTION: OFFER {p:g}")
            out.append(Generation(text=text))
        return out
