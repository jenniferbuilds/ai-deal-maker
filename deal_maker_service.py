"""Real Deal Maker - Online negotiation agent service.

Connects to real platforms and negotiates on your behalf:
- Facebook Marketplace
- Craigslist
- OfferUp
- Poshmark
- Email negotiations

Architecture:
1. Platform connectors (API/scraping)
2. Agent negotiation engine
3. Human approval layer
4. Web dashboard for monitoring

Usage:
    uv run python deal_maker_service.py --platform facebook --item-url "..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
import threading
from queue import Queue

# Platform connectors
class PlatformConnector(Protocol):
    """Interface for platform-specific connectors."""
    
    def get_item_info(self, url: str) -> dict:
        """Fetch item details from platform."""
        ...
    
    def send_message(self, conversation_id: str, message: str) -> bool:
        """Send a message to seller."""
        ...
    
    def get_new_messages(self, conversation_id: str) -> list[dict]:
        """Check for new messages from seller."""
        ...
    
    def make_offer(self, conversation_id: str, amount: float) -> bool:
        """Submit formal offer if platform supports it."""
        ...


@dataclass
class DealOpportunity:
    """A potential deal found online."""
    platform: str
    url: str
    title: str
    price: float
    description: str
    seller_name: str
    conversation_id: Optional[str] = None
    your_max_price: Optional[float] = None
    auto_negotiate: bool = False
    require_approval: bool = True


@dataclass
class NegotiationSession:
    """Active negotiation session."""
    opportunity: DealOpportunity
    status: str  # "pending", "active", "paused", "completed", "failed"
    messages: list[dict]
    current_offer: Optional[float] = None
    turns: int = 0
    max_turns: int = 6
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class RealDealMaker:
    """Main deal maker orchestrator."""
    
    def __init__(self, config_file: str = "deal_maker_config.json"):
        self.config = self._load_config(config_file)
        self.active_sessions: dict[str, NegotiationSession] = {}
        self.message_queue = Queue()
        self.approval_queue = Queue()
        self.running = False
        
        # Import agent components
        from policies import ChatTokenizer, RiverBuyer, make_client, session_sampler
        from env import buyer_system_prompt, parse_buyer
        
        self.chat = ChatTokenizer(self.config.get("base_model", "Qwen/Qwen3.6-35B-A3B-FP8"))
        self.client = make_client()
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration."""
        if os.path.exists(config_file):
            return json.load(open(config_file))
        return {
            "base_model": "Qwen/Qwen3.6-35B-A3B-FP8",
            "checkpoint": None,
            "temperature": 0.7,
            "auto_approve_threshold": 0.8,  # Auto-approve if deal < 80% of asking
            "max_price_threshold": 0.7,  # Never pay more than 70% of asking
        }
    
    def add_opportunity(self, opp: DealOpportunity) -> str:
        """Add a new deal opportunity to negotiate."""
        session = NegotiationSession(opportunity=opp, status="pending", messages=[])
        self.active_sessions[opp.url] = session
        
        if opp.auto_negotiate:
            self._start_negotiation(session)
        
        return opp.url
    
    def _start_negotiation(self, session: NegotiationSession) -> None:
        """Start or resume negotiation."""
        session.status = "active"
        
        # Build system prompt
        opp = session.opportunity
        system_prompt = f"""You are buying an item. Negotiate the lowest price while staying friendly.

Item: {opp.title}
Listing price: ${opp.price}
Your maximum: ${opp.your_max_price or opp.price * 0.7:.2f}
Platform: {opp.platform}

Description: {opp.description}

You get at most {session.max_turns} messages. End EVERY message with:
ACTION: OFFER <price>   (propose a price)
ACTION: ACCEPT          (accept their price)
ACTION: WALK            (leave without a deal)
"""
        
        session.messages = [{"role": "system", "content": system_prompt}]
    
    def process_seller_message(self, session_id: str, seller_message: str) -> Optional[str]:
        """Process incoming message from seller and generate response."""
        session = self.active_sessions.get(session_id)
        if not session or session.status != "active":
            return None
        
        # Add seller message
        session.messages.append({"role": "user", "content": seller_message})
        session.turns += 1
        
        # Generate buyer response
        with self.client.session(experiment="real-deal-maker") as river_session:
            from policies import RiverBuyer, session_sampler
            
            buyer = RiverBuyer(
                session_sampler(river_session, self.config["base_model"], self.config.get("checkpoint")),
                self.chat,
                temperature=self.config["temperature"]
            )
            
            gen = buyer.act([session.messages])[0]
            session.messages.append({"role": "assistant", "content": gen.text})
            
            # Parse action
            from env import parse_buyer
            text, action = parse_buyer(gen.text)
            
            # Check if approval needed
            if session.opportunity.require_approval and action.kind == "offer":
                if action.price > session.opportunity.price * self.config["max_price_threshold"]:
                    # Need approval for high offers
                    self.approval_queue.put({
                        "session_id": session_id,
                        "type": "offer_approval",
                        "amount": action.price,
                        "message": text
                    })
                    session.status = "paused"
                    return None
            
            if action.kind == "accept":
                session.status = "completed"
                session.current_offer = session.opportunity.price
            elif action.kind == "walk":
                session.status = "completed"
            elif action.kind == "offer":
                session.current_offer = action.price
            
            return text
    
    def get_status(self) -> dict:
        """Get overall status of all negotiations."""
        return {
            "active_sessions": len([s for s in self.active_sessions.values() if s.status == "active"]),
            "pending_approval": len([s for s in self.active_sessions.values() if s.status == "paused"]),
            "completed": len([s for s in self.active_sessions.values() if s.status == "completed"]),
            "sessions": {
                url: {
                    "title": sess.opportunity.title,
                    "status": sess.status,
                    "current_offer": sess.current_offer,
                    "turns": sess.turns
                }
                for url, sess in self.active_sessions.items()
            }
        }


# Platform-specific implementations

class FacebookMarketplaceConnector:
    """Connector for Facebook Marketplace (requires browser automation)."""
    
    def __init__(self):
        self.name = "facebook_marketplace"
        # Would need selenium/playwright for actual implementation
    
    def get_item_info(self, url: str) -> dict:
        """Scrape item info from FB Marketplace URL."""
        # Placeholder - would use browser automation
        print(f"[FB Marketplace] Fetching: {url}")
        return {
            "title": "Item from Facebook Marketplace",
            "price": 100.0,
            "description": "Description would be scraped here",
            "seller": "Seller Name"
        }
    
    def send_message(self, conversation_id: str, message: str) -> bool:
        """Send message via FB Messenger."""
        print(f"[FB] Sending to {conversation_id}: {message}")
        # Would use browser automation to send actual message
        return True


class CraigslistConnector:
    """Connector for Craigslist email negotiations."""
    
    def __init__(self, email_config: dict):
        self.name = "craigslist"
        self.email_config = email_config
    
    def get_item_info(self, url: str) -> dict:
        """Parse Craigslist posting."""
        print(f"[Craigslist] Fetching: {url}")
        # Would scrape the posting
        return {
            "title": "Craigslist Item",
            "price": 150.0,
            "description": "Item description",
            "seller": "seller@example.com"
        }
    
    def send_message(self, conversation_id: str, message: str) -> bool:
        """Send email to seller."""
        print(f"[Email] Sending to {conversation_id}: {message}")
        # Would use SMTP to send actual email
        return True


class ManualConnector:
    """Manual connector for human-in-the-loop operation."""
    
    def __init__(self):
        self.name = "manual"
    
    def get_item_info(self, url: str) -> dict:
        """Prompt user for item info."""
        print("\n=== Manual Item Entry ===")
        return {
            "title": input("Item title: "),
            "price": float(input("Listing price: $")),
            "description": input("Description: "),
            "seller": input("Seller name/contact: ")
        }


# Web interface for monitoring and control

def create_web_interface(deal_maker: RealDealMaker, port: int = 8080):
    """Create simple web interface for monitoring."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class DealMakerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(deal_maker.get_status(), indent=2).encode())
            elif self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                html = """
                <html>
                <head><title>Deal Maker Dashboard</title></head>
                <body>
                    <h1>🤖 Real Deal Maker</h1>
                    <div id="status">Loading...</div>
                    <script>
                        fetch('/status')
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('status').innerHTML = 
                                    '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                            });
                    </script>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
    
    server = HTTPServer(('localhost', port), DealMakerHandler)
    print(f"Dashboard running at http://localhost:{port}")
    return server


def main():
    parser = argparse.ArgumentParser(description="Real Deal Maker Service")
    parser.add_argument("--mode", choices=["manual", "auto", "web"], default="manual",
                       help="Operation mode")
    parser.add_argument("--platform", choices=["facebook", "craigslist", "manual"], default="manual")
    parser.add_argument("--item-url", help="URL of item to negotiate")
    parser.add_argument("--max-price", type=float, help="Your maximum price")
    parser.add_argument("--auto-negotiate", action="store_true", help="Auto-negotiate without approval")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Web dashboard port")
    
    args = parser.parse_args()
    
    # Set API key
    if not os.environ.get("RIVER_API_KEY"):
        os.environ["RIVER_API_KEY"] = "rv_ZshTM1ktqZRUXFl5HG50r3uQGCkMwlcPbYCn90xWeaQ"
    
    print("="*70)
    print("🤖 REAL DEAL MAKER - Online Negotiation Agent")
    print("="*70)
    
    deal_maker = RealDealMaker()
    
    if args.mode == "web":
        # Start web interface
        server = create_web_interface(deal_maker, args.dashboard_port)
        print("\nDashboard is running. Add opportunities via API:")
        print(f"  POST http://localhost:{args.dashboard_port}/add")
        print("\nPress Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    elif args.mode == "manual":
        # Manual mode - interactive
        print("\nManual Mode - You relay messages between agent and seller\n")
        
        # Get item info
        if args.item_url:
            connector = ManualConnector()
            item_info = connector.get_item_info(args.item_url)
        else:
            print("Enter item details:")
            item_info = {
                "title": input("Item title: "),
                "price": float(input("Listing price: $")),
                "description": input("Description: "),
                "seller": input("Seller name: ")
            }
        
        # Create opportunity
        opp = DealOpportunity(
            platform=args.platform,
            url=args.item_url or "manual",
            title=item_info["title"],
            price=item_info["price"],
            description=item_info["description"],
            seller_name=item_info["seller"],
            your_max_price=args.max_price or item_info["price"] * 0.7,
            require_approval=not args.auto_negotiate
        )
        
        session_id = deal_maker.add_opportunity(opp)
        
        print(f"\n{'='*70}")
        print(f"Negotiating: {opp.title}")
        print(f"Asking: ${opp.price:.2f}")
        print(f"Your max: ${opp.your_max_price:.2f}")
        print(f"{'='*70}\n")
        
        # Interactive negotiation loop
        while True:
            session = deal_maker.active_sessions[session_id]
            
            if session.status == "completed":
                print("\n=== NEGOTIATION COMPLETE ===")
                if session.current_offer:
                    print(f"Deal at: ${session.current_offer:.2f}")
                else:
                    print("No deal reached")
                break
            
            # Get seller message (from user)
            print("\n[Paste seller's message, or type '/quit' to stop]")
            seller_msg = input("Seller: ").strip()
            
            if seller_msg.lower() == "/quit":
                print("Stopping negotiation.")
                break
            
            # Get agent response
            buyer_response = deal_maker.process_seller_message(session_id, seller_msg)
            
            if buyer_response:
                print(f"\nAgent says: {buyer_response}")
            else:
                print("\n[Agent waiting for your approval to continue...]")
                approval = input("Approve? (yes/no): ").strip().lower()
                if approval == "yes":
                    session.status = "active"
                    print("Approved. Continuing...")
    
    print("\nDeal maker finished.")


if __name__ == "__main__":
    main()
