#!/bin/bash

# AI Deal Maker Screen Recording Script for macOS

echo "🎬 AI Deal Maker - Screen Recording Script"
echo "============================================"
echo ""

# Check if dashboard is running
if ! curl -s http://localhost:8080/status > /dev/null 2>&1; then
    echo "❌ Dashboard not running. Please start it first:"
    echo "   uv run python api_server.py --port 8080"
    exit 1
fi

echo "✅ Dashboard is running at http://localhost:8080"
echo ""

# Create recordings directory
mkdir -p recordings

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="recordings/demo_$TIMESTAMP.mov"

echo "📹 Recording options:"
echo ""
echo "1. Record full screen with QuickTime (manual)"
echo "2. Capture screenshot of dashboard"
echo "3. Create automated demo video with screenshots"
echo ""
read -p "Select option [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🎬 Opening QuickTime Player for manual recording..."
        echo ""
        echo "Instructions:"
        echo "1. QuickTime will open"
        echo "2. File → New Screen Recording"
        echo "3. Select the browser window with dashboard"
        echo "4. Click 'Record'"
        echo "5. Navigate to http://localhost:8080 if not already open"
        echo "6. Stop recording when done"
        echo ""
        open -a QuickTime\ Player
        ;;
        
    2)
        echo ""
        echo "📸 Capturing screenshot..."
        screencapture -x "$OUTPUT_FILE.jpg"
        echo "✅ Screenshot saved to: $OUTPUT_FILE.jpg"
        open "$OUTPUT_FILE.jpg"
        ;;
        
    3)
        echo ""
        echo "🎬 Creating automated demo..."
        echo ""
        
        # Capture multiple screenshots for demo
        echo "📸 Capturing dashboard..."
        open "http://localhost:8080"
        sleep 2
        
        for i in {1..5}; do
            screencapture -x "recordings/frame_$TIMESTAMP\_$i.png"
            echo "  Frame $i captured"
            sleep 2
        done
        
        echo ""
        echo "✅ Captured 5 frames to recordings/"
        echo ""
        echo "To create a video from frames, run:"
        echo "  ffmpeg -framerate 1 -i recordings/frame_${TIMESTAMP}_%d.png -c:v libx264 $OUTPUT_FILE"
        echo ""
        echo "Or use QuickTime to record manually for better quality."
        ;;
        
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "📊 Dashboard URL: http://localhost:8080"
echo "📁 Recordings saved to: recordings/"
