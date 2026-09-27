#!/bin/bash
echo "🎬 Creating Quick Video Demo"
echo "============================"
echo ""

# Check dashboard
if ! curl -s http://localhost:8080/status > /dev/null 2>&1; then
    echo "❌ Dashboard not running"
    exit 1
fi

echo "✅ Dashboard is running"
echo ""
echo "📸 Taking screenshots..."

# Open dashboard
open "http://localhost:8080"
sleep 2

# Capture frames
for i in {1..10}; do
    screencapture -x "video/frame_$(printf %03d $i).png"
    echo "  Frame $i captured"
    sleep 1
done

echo ""
echo "✅ Captured 10 frames"
echo ""
echo "📹 To create a video:"
echo "   1. Open QuickTime Player"
echo "   2. File → Open Image Sequence"
echo "   3. Select video/frame_001.png"
echo "   4. Set frame rate to 1 fps"
echo "   5. Export as movie"
echo ""
echo "Or install ffmpeg:"
echo "   brew install ffmpeg"
echo "   ffmpeg -framerate 1 -i video/frame_%03d.png -c:v libx264 video/demo.mp4"
