#!/usr/bin/env python3
"""
Simple test of deterministic composition with existing character images.
"""

from PIL import Image, ImageDraw
from pathlib import Path

# Use existing character references
char1_path = "/Users/edgarberlinck/code/story-engine/outputs/Test_ui/characters/Nikita/reference.png"
char2_path = "/Users/edgarberlinck/code/story-engine/outputs/Test_ui/characters/Roger/reference.png"
output_dir = "/tmp/scene_test"

Path(output_dir).mkdir(parents=True, exist_ok=True)

print("Testing simple composition...")

# Create background
bg = Image.new('RGB', (1024, 1024), color=(139, 69, 19))  # Brown
draw = ImageDraw.Draw(bg)
draw.rectangle([0, 800, 1024, 1024], fill=(101, 67, 33))  # Stage floor

# Load characters (fallback to colored rectangles if images fail)
def load_or_dummy(path, color):
    try:
        img = Image.open(path).convert('RGBA')
        print(f"Loaded {path}: {img.size}")
        return img
    except Exception as e:
        print(f"Could not load {path}: {e}, using dummy")
        return Image.new('RGBA', (300, 500), color)

char1 = load_or_dummy(char1_path, (220, 100, 100, 255))
char2 = load_or_dummy(char2_path, (100, 100, 220, 255))

# Resize and place
def place_char(bg, char_img, anchor_x, anchor_y, scale=0.4):
    target_h = int(bg.height * scale)
    aspect = char_img.width / char_img.height
    target_w = int(target_h * aspect)
    
    resized = char_img.resize((target_w, target_h), Image.LANCZOS)
    
    x = int(anchor_x * bg.width) - target_w // 2
    y = int(anchor_y * bg.height) - target_h
    
    bg.paste(resized, (x, y), resized)
    print(f"Placed character at ({x}, {y}), size {target_w}x{target_h}")

place_char(bg, char1, 0.3, 0.85, scale=0.5)
place_char(bg, char2, 0.7, 0.8, scale=0.45)

output_path = f"{output_dir}/composed_scene.png"
bg.convert('RGB').save(output_path)
print(f"\nSaved composed scene to {output_path}")
print("Composition test complete!")
