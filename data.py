"""CraigslistBargain loader.

Downloads the raw parsed.json files (the same URLs the Hugging Face
`stanfordnlp/craigslist_bargains` loader script uses), caches them under
data/, and turns each unique posting scenario into a buyer-side Scenario.

Dataset: He et al. 2018, "Decoupling Strategy and Generation in
Negotiation Dialogues" (https://arxiv.org/abs/1808.09637).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

URLS = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
}

# The seller's hidden floor is a deterministic fraction of the listing
# price, drawn per scenario so the agent can't memorize a single number.
FLOOR_MIN_FRAC = 0.60
FLOOR_MAX_FRAC = 0.85
MAX_DESCRIPTION_CHARS = 600


@dataclass(frozen=True)
class Scenario:
    uid: str
    title: str
    category: str
    description: str
    listing_price: float
    buyer_target: float
    floor: float  # seller's secret minimum; never shown to the buyer
    human_prices: tuple[float, ...]  # final prices humans agreed on (reference only)


def _download(split: str) -> list[dict]:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{split}.json")
    if not os.path.exists(path):
        print(f"Downloading CraigslistBargain {split} split ...")
        tmp = path + ".tmp"
        with urllib.request.urlopen(URLS[split], timeout=120) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        os.replace(tmp, path)
    with open(path) as f:
        return json.load(f)


def _floor_for(uid: str, listing: float) -> float:
    h = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    frac = FLOOR_MIN_FRAC + h * (FLOOR_MAX_FRAC - FLOOR_MIN_FRAC)
    return round_price(listing * frac, listing)


def round_price(value: float, listing: float) -> float:
    """Whole dollars for normal items, cents for very cheap ones."""
    return round(value, 2) if listing < 20 else float(round(value))


def load_scenarios(split: str) -> list[Scenario]:
    dialogues = _download(split)
    by_uid: dict[str, dict] = {}
    human: dict[str, list[float]] = {}

    for d in dialogues:
        scenario = d.get("scenario") or {}
        uid = d.get("scenario_uuid") or scenario.get("uuid")
        kbs = scenario.get("kbs") or []
        buyer_kb = next((kb for kb in kbs if kb.get("personal", {}).get("Role") == "buyer"), None)
        if not uid or buyer_kb is None:
            continue
        item = buyer_kb.get("item", {})
        listing = item.get("Price")
        target = buyer_kb["personal"].get("Target")
        if not listing or listing <= 0 or not target or target <= 0:
            continue

        outcome = d.get("outcome") or {}
        offer = outcome.get("offer") or {}
        if outcome.get("reward") == 1 and offer.get("price"):
            human.setdefault(uid, []).append(float(offer["price"]))

        if uid not in by_uid:
            desc = " ".join(item.get("Description") or []).strip()
            by_uid[uid] = {
                "title": (item.get("Title") or "").strip(),
                "category": item.get("Category") or scenario.get("category") or "",
                "description": desc[:MAX_DESCRIPTION_CHARS],
                "listing": float(listing),
                "target": float(target),
            }

    scenarios = [
        Scenario(
            uid=uid,
            title=v["title"],
            category=v["category"],
            description=v["description"],
            listing_price=v["listing"],
            buyer_target=v["target"],
            floor=_floor_for(uid, v["listing"]),
            human_prices=tuple(human.get(uid, [])),
        )
        for uid, v in by_uid.items()
    ]
    scenarios.sort(key=lambda s: s.uid)  # stable order across runs
    return scenarios


if __name__ == "__main__":
    for split in URLS:
        scs = load_scenarios(split)
        with_human = [s for s in scs if s.human_prices]
        ratio = sum(s.human_prices[0] / s.listing_price for s in with_human) / max(1, len(with_human))
        print(f"{split}: {len(scs)} scenarios, human deals on {len(with_human)}, "
              f"mean human price = {ratio:.1%} of listing")
        print("  e.g.", scs[0])
