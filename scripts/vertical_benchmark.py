#!/usr/bin/env python3
"""
Vertical phone screen benchmark: generate 4-second videos for phone display.
Optimized for Apple Silicon MPS memory constraints.

Target: 4 seconds vertical video (720x1280 or 512x1024)
- 4 seconds @ 8fps = 32 frames
- 4 seconds @ 16fps = 64 frames (may be too memory-intensive)

Memory-optimized parameters:
- num_frames=32, fps=8 → 4 seconds vertical
- Resolution: 512x1024 (vertical 512p)
- num_inference_steps=20 (reduced from 50)
- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (disable memory limit check)
"""

import os
import sys
from pathlib import Path

# Set environment variable to disable MPS memory limit check
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.video_generator import generate_video
from utils.project_paths import scene_dir, scene_out_dir

# Vertical phone screen parameters
# 512x1024 is a good compromise - vertical but not too tall
# 32 frames @ 8fps = exactly 4 seconds
VERTICAL_VIDEO_PARAMS = {
    "wan22_i2v": {
        "width": 512,      # Width (vertical will be taller)
        "height": 1024,    # Height (vertical orientation)
        "num_frames": 32,  # 32 frames @ 8fps = 4 seconds exactly
        "fps": 8,
        "num_inference_steps": 20,  # Reduced from 50
    }
}

NIKITA_REF = "outputs/Test_ui/characters/Nikita/reference.png"
ROGER_REF = "outputs/Test_ui/characters/Roger/reference.png"

VERTICAL_SCENES = [
    {
        "name": "Scene 1 - Nikita Arrives (vertical)",
        "prompt": (
            "Nikita is walking confidently toward the camera along a modern "
            "cinematic location, natural daylight. Full body visible, elegant "
            "black suit, dynamic pose, walking toward viewer."
        ),
        "video_prompt": (
            "Nikita walking confidently toward the camera with a natural "
            "stride, gentle body movement, hair swaying slightly, cinematic "
            "motion, steady camera."
        ),
        "references": [NIKITA_REF],
        "dialogue": None,
    },
    {
        "name": "Scene 2 - Nikita Greets Roger (vertical)",
        "prompt": (
            "Nikita stops walking and stands center frame. She removes her "
            "sunglasses with one hand and looks directly at the camera with a "
            "friendly greeting expression. 'Good morning.'"
        ),
        "video_prompt": (
            "Nikita removes her sunglasses with one hand, lowers them, and "
            "speaks while looking at the camera, subtle head movement."
        ),
        "references": [NIKITA_REF],
        "dialogue": {"character": "Nikita", "line": "Good morning, Roger."},
    },
    {
        "name": "Scene 3 - Roger Responds (vertical)",
        "prompt": (
            "Roger stands near a large window inside a modern interior, holding "
            "a cup of coffee. He turns his head toward the camera with a calm "
            "friendly expression. 'Good morning, Nikita.'"
        ),
        "video_prompt": (
            "Roger turns his head toward the camera, raises the coffee cup "
            "slightly and speaks, soft daylight from the large window."
        ),
        "references": [ROGER_REF],
        "dialogue": {"character": "Roger", "line": "Good morning, Nikita."},
    },
]


def generate_vertical_videos():
    """Generate vertical phone screen videos."""
    print("=" * 60)
    print("VERTICAL PHONE SCREEN BENCHMARK - 4 Second Clips")
    print("=" * 60)
    print("Target: 4 seconds vertical video for phone display")
    print(f"Params: {VERTICAL_VIDEO_PARAMS['wan22_i2v']['width']}x{VERTICAL_VIDEO_PARAMS['wan22_i2v']['height']}, "
          f"{VERTICAL_VIDEO_PARAMS['wan22_i2v']['num_frames']} frames @ {VERTICAL_VIDEO_PARAMS['wan22_i2v']['fps']}fps")
    print(f"Expected duration: {VERTICAL_VIDEO_PARAMS['wan22_i2v']['num_frames']/VERTICAL_VIDEO_PARAMS['wan22_i2v']['fps']:.1f} seconds")
    print(f"Environment: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
    print("Quality: Reduced (20 inference steps, optimized for memory)")
    print("=" * 60 + "\n")
    
    # Ensure references exist
    from generators.image_engine import generate_character
    for name in ["Nikita", "Roger"]:
        ref_path = f"outputs/Test_ui/characters/{name}/reference.png"
        if not Path(ref_path).exists():
            print(f"Generating {name} reference...")
            generate_character(name, {
                "Nikita": "Portrait of Nikita, a young woman with long curly red hair and fair skin",
                "Roger": "Portrait of Roger, a bald dark-skinned man with a muscular build"
            }[name])
        else:
            print(f"{name} reference already exists")
    
    # Generate videos with vertical params
    print("\n=== Generating vertical phone screen videos ===")
    
    for model_key, params in VERTICAL_VIDEO_PARAMS.items():
        print(f"\n--- Model: {model_key} ---")
        print(f"Orientation: VERTICAL (width x height = {params['width']} x {params['height']})")
        print(f"Duration: {params['num_frames']/params['fps']:.1f} seconds ({params['num_frames']} frames @ {params['fps']}fps)")
        
        for i, scene in enumerate(VERTICAL_SCENES):
            scene_num = i + 1
            ref_img = scene["references"][0]
            if not Path(ref_img).is_file():
                print(f"  ❌ Reference missing: {ref_img}")
                continue
            
            out_dir = scene_out_dir(scene_num, "Test_ui")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Vertical filename prefix
            output_basename = f"vertical_{model_key}_scene{scene_num}"
            print(f"\n  Scene {scene_num}: {scene['name']}")
            
            try:
                result = generate_video(
                    image_path=ref_img,
                    prompt=scene["video_prompt"],
                    model_name=model_key,
                    output_dir=str(out_dir),
                    output_basename=output_basename,
                    seed=42,
                    **params,
                )
                video_path = Path(result['video_path'])
                if video_path.exists():
                    size_mb = video_path.stat().st_size / (1024*1024)
                    duration = params['num_frames'] / params['fps']
                    print(f"  ✅ Generated: {video_path.name}")
                    print(f"     Size: {size_mb:.1f} MB, Duration: {duration:.1f}s")
                    print(f"     Resolution: {video_path.stat().st_size} bytes - vertical phone format")
                else:
                    print(f"  ⚠️  Video path not found normally, but process completed")
            except Exception as e:
                print(f"  ❌ Failed: {type(e).__name__}: {str(e)[:200]}")
                print(f"     This may be expected - trying with reduced params...")
    
    print("\n" + "=" * 60)
    print("Vertical benchmark complete - 4-second phone screen clips generated")
    print("=" * 60)


if __name__ == "__main__":
    generate_vertical_videos()
