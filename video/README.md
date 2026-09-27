# 🎬 Video Demo Instructions

## Quick Method (Automated)

1. ✅ **Screenshots captured**: 10 frames in `video/frame_001.png` to `frame_010.png`

2. **Create video with QuickTime**:
   ```
   Open QuickTime Player
   File → Open Image Sequence
   Select: frame_001.png
   Frame rate: 1 fps
   File → Export As: "AI_Deal_Maker_Demo.mov"
   ```

3. **Or install ffmpeg** (faster):
   ```bash
   brew install ffmpeg
   cd negotiation-agent
   ffmpeg -framerate 1 -i video/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 video/demo.mp4
   ```

## What's Captured

✅ **Dashboard screenshot** showing:
- Training progress: Step 39/50 (78% complete)
- Perfect reward: +1.00
- 100% deal rate
- 30.5% average discount
- Real-time metrics
- Feature overview

## Dashboard URL

🌐 http://localhost:8080

## Training Metrics

| Metric | Value |
|--------|-------|
| Current Step | 39/50 |
| Reward | +1.00 (perfect) |
| Deal Rate | 100% |
| Price vs Listing | 69.5% (30.5% discount) |
| Rudeness | 0% |
| Format Errors | 0% |

## Next Steps

1. Open dashboard: http://localhost:8080
2. Use QuickTime to create video from screenshots
3. Or install ffmpeg for automated video creation

---

**Video will show**: AI-powered negotiation agent dashboard with real-time training progress and impressive performance metrics!
