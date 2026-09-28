"""Local inference without River cloud.

Options:
1. Use River API (default - requires API key, runs on cloud)
2. Use local Ollama models (free, runs locally)
3. Use HuggingFace models (free, runs locally with GPU)

Usage:
    # With River (cloud)
    uv run python local_inference.py --mode river
    
    # With Ollama (local)
    ollama pull llama3.1:8b
    uv run python local_inference.py --mode ollama
    
    # With HuggingFace (local, needs GPU)
    uv run python local_inference.py --mode hf --model "Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import os
import json

def negotiate_with_river(item: str, price: float, target: float):
    """Use River cloud for inference."""
    os.environ.setdefault("RIVER_API_KEY", "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ")
    
    from policies import ChatTokenizer, RiverBuyer, make_client, session_sampler, DEFAULT_BASE_MODEL
    from env import buyer_system_prompt, parse_buyer
    
    # Load trained checkpoint
    checkpoint = None
    try:
        with open("runs/v2/checkpoints.json") as f:
            checkpoint = json.load(f).get("latest")
    except:
        pass
    
    chat = ChatTokenizer(DEFAULT_BASE_MODEL)
    client = make_client()
    
    class QuickScenario:
        uid = "local"
        title = item
        category = "general"
        description = item
        listing_price = price
        buyer_target = target
        floor = target * 0.9
    
    with client.session(experiment="local-inference") as session:
        buyer = RiverBuyer(
            session_sampler(session, DEFAULT_BASE_MODEL, checkpoint),
            chat, temperature=0.7
        )
        
        messages = [{"role": "system", "content": buyer_system_prompt(QuickScenario(), 6)}]
        messages.append({"role": "user", "content": f"Hi! I'm selling {item}. Asking ${price}."})
        
        gen = buyer.act([messages])[0]
        text, action = parse_buyer(gen.text)
        
        return text, action


def negotiate_with_ollama(item: str, price: float, target: float, model: str = "llama3.1:8b"):
    """Use local Ollama model for inference."""
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        return None, None
    
    system_prompt = f"""You are negotiating to buy an item. Be friendly but firm.
Item: {item}
Listing price: ${price}
Your target: ${target}
Your maximum: ${price * 0.85}

Negotiate the lowest price. End EVERY message with:
ACTION: OFFER <price>  (to make an offer)
ACTION: ACCEPT         (to accept their price)
ACTION: WALK           (to walk away)"""

    user_msg = f"Hi! I'm selling {item}. Asking ${price}."
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("message", {}).get("content", "")
            
            # Parse action
            from env import parse_buyer
            clean_text, action = parse_buyer(text)
            return clean_text, action
        else:
            print(f"Ollama error: {response.status_code}")
            return None, None
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running. Start it with: ollama serve")
        print("   Then pull a model: ollama pull llama3.1:8b")
        return None, None


def negotiate_with_hf(item: str, price: float, target: float, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Use local HuggingFace model for inference."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Install transformers: pip install transformers torch")
        return None, None
    
    print(f"Loading model {model_name}... (this may take a while)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    system_prompt = f"""You are negotiating to buy an item. Be friendly but firm.
Item: {item}
Listing price: ${price}
Your target: ${target}

End EVERY message with: ACTION: OFFER <price> or ACTION: ACCEPT or ACTION: WALK"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hi! I'm selling {item}. Asking ${price}."}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    response = response.split("assistant")[-1].strip()
    
    from env import parse_buyer
    clean_text, action = parse_buyer(response)
    return clean_text, action


def main():
    parser = argparse.ArgumentParser(description="Local negotiation inference")
    parser.add_argument("--mode", choices=["river", "ollama", "hf"], default="river",
                       help="Inference mode: river (cloud), ollama (local), hf (huggingface)")
    parser.add_argument("--model", default="llama3.1:8b", help="Model name for ollama/hf")
    parser.add_argument("--item", default="iPhone 13 Pro", help="Item to negotiate")
    parser.add_argument("--price", type=float, default=500, help="Listing price")
    parser.add_argument("--target", type=float, default=350, help="Target price")
    
    args = parser.parse_args()
    
    print(f"\n🤖 AI Deal Maker - Local Inference")
    print(f"=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Item: {args.item}")
    print(f"Listing: ${args.price}")
    print(f"Target: ${args.target}")
    print(f"=" * 50)
    
    if args.mode == "river":
        text, action = negotiate_with_river(args.item, args.price, args.target)
    elif args.mode == "ollama":
        text, action = negotiate_with_ollama(args.item, args.price, args.target, args.model)
    else:  # hf
        text, action = negotiate_with_hf(args.item, args.price, args.target, args.model)
    
    if text:
        print(f"\n🤖 AGENT: {text}")
        if action and action.kind == "offer":
            print(f"   [OFFERS: ${action.price}]")
        elif action:
            print(f"   [ACTION: {action.kind.upper()}]")
    else:
        print("\n❌ No response generated")
    
    print(f"\n" + "=" * 50)
    print("📋 Local Inference Options:")
    print("   1. River API (cloud, uses trained model)")
    print("   2. Ollama (local, free, needs: ollama pull llama3.1:8b)")
    print("   3. HuggingFace (local, free, needs GPU for best performance)")


if __name__ == "__main__":
    main()
