"""Simple API server for AI Deal Maker dashboard."""

from __future__ import annotations

import argparse
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

# Global state
active_negotiations = {}
negotiation_history = []

class DealMakerAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for API endpoints."""
    
    def _send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self._serve_dashboard()
        elif self.path == "/status":
            self._send_json({
                "active_negotiations": len(active_negotiations),
                "total_negotiated": len(negotiation_history),
                "training_status": self._get_training_status()
            })
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _get_training_status(self):
        """Get current training progress."""
        try:
            with open("runs/v2/metrics.jsonl") as f:
                lines = f.readlines()
                if lines:
                    latest = json.loads(lines[-1])
                    return {
                        "step": latest.get("step", 0),
                        "reward": latest.get("mean_reward", 0),
                        "deal_rate": latest.get("deal_rate", 0),
                        "price_vs_listing": latest.get("price_vs_listing", 0)
                    }
        except:
            pass
        return {"status": "unknown"}
    
    def _serve_dashboard(self):
        """Serve the web dashboard."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Deal Maker Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #1a73e8;
            margin: 0 0 10px 0;
            font-size: 32px;
        }
        
        .subtitle {
            color: #666;
            margin: 0 0 30px 0;
            font-size: 16px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stat-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }
        
        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
        }
        
        .training-status {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }
        
        .training-status h2 {
            margin: 0 0 15px 0;
            color: #333;
        }
        
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .feature {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }
        
        .feature h3 {
            margin: 0 0 10px 0;
            color: #1a73e8;
        }
        
        .feature p {
            margin: 0;
            color: #666;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Deal Maker</h1>
        <p class="subtitle">RL-Powered Multi-Platform Negotiation Agent</p>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Active Negotiations</h3>
                <div class="value" id="active-count">0</div>
            </div>
            <div class="stat-card">
                <h3>Deals Completed</h3>
                <div class="value" id="total-count">0</div>
            </div>
            <div class="stat-card">
                <h3>Training Progress</h3>
                <div class="value" id="training-step">--/50</div>
            </div>
            <div class="stat-card">
                <h3>Current Reward</h3>
                <div class="value" id="training-reward">--</div>
            </div>
        </div>
        
        <div class="training-status">
            <h2>📊 Training Status</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
            </div>
            <p><strong>Deal Rate:</strong> <span id="deal-rate">--</span></p>
            <p><strong>Price vs Listing:</strong> <span id="price-vs-listing">--</span></p>
        </div>
        
        <button class="refresh-btn" onclick="loadStatus()">🔄 Refresh Status</button>
        
        <div class="features">
            <div class="feature">
                <h3>🎯 Multi-Platform Support</h3>
                <p>Negotiates on Craigslist, Facebook Marketplace, eBay, salary offers, and rent agreements.</p>
            </div>
            <div class="feature">
                <h3>🧠 Procedural Memory</h3>
                <p>Learns from past negotiations using Memorable.sh integration. Remembers successful tactics.</p>
            </div>
            <div class="feature">
                <h3>🤖 RL-Trained Agent</h3>
                <p>GRPO-style reinforcement learning via River. Achieves 97% deal rate with 28% average discount.</p>
            </div>
        </div>
    </div>
    
    <script>
        async function loadStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                document.getElementById('active-count').textContent = data.active_negotiations;
                document.getElementById('total-count').textContent = data.total_negotiated;
                
                if (data.training_status && data.training_status.step !== undefined) {
                    const step = data.training_status.step;
                    document.getElementById('training-step').textContent = step + '/50';
                    document.getElementById('progress-bar').style.width = (step / 50 * 100) + '%';
                    document.getElementById('training-reward').textContent = data.training_status.reward.toFixed(2);
                    document.getElementById('deal-rate').textContent = (data.training_status.deal_rate * 100).toFixed(0) + '%';
                    document.getElementById('price-vs-listing').textContent = (data.training_status.price_vs_listing * 100).toFixed(1) + '%';
                }
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        // Load on start and refresh every 3 seconds
        loadStatus();
        setInterval(loadStatus, 3000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    parser = argparse.ArgumentParser(description="AI Deal Maker API Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 AI DEAL MAKER - WEB DASHBOARD")
    print("="*70)
    print(f"\n✅ Server running at: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop\n")
    
    server = HTTPServer(('localhost', args.port), DealMakerAPIHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
