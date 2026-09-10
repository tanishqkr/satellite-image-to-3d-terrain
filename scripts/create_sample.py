"""Generate a sample aerial RGB image for testing."""
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def create_sample_scene():
    width, height = 512, 512
    # Base terrain (greenish/earthy ground)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = [70, 95, 60]

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Roads (dark grey)
    draw.rectangle([0, 240, 512, 272], fill=(45, 45, 48))
    draw.rectangle([240, 0, 272, 512], fill=(45, 45, 48))

    # Road centerlines (yellow/white dashes)
    for x in range(0, 512, 30):
        draw.line([(x, 256), (x + 15, 256)], fill=(220, 200, 50), width=2)
        draw.line([(256, x), (256, x + 15)], fill=(220, 200, 50), width=2)

    # Buildings (various roofs)
    # Block 1 (top-left): Large warehouse
    draw.rectangle([40, 40, 180, 160], fill=(180, 185, 190), outline=(100, 100, 105), width=2)
    draw.rectangle([60, 60, 160, 140], fill=(160, 165, 170))

    # Block 2 (top-right): Office towers
    draw.rectangle([320, 50, 440, 170], fill=(70, 90, 130), outline=(40, 50, 80), width=3)
    draw.rectangle([350, 80, 410, 140], fill=(110, 130, 170))

    # Block 3 (bottom-left): Residential houses with red/brown roofs
    houses = [
        (40, 310, 100, 370, (160, 60, 50)),
        (130, 310, 190, 370, (140, 70, 60)),
        (40, 400, 100, 460, (150, 65, 55)),
        (130, 400, 190, 460, (170, 75, 65)),
    ]
    for x1, y1, x2, y2, color in houses:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(90, 40, 30), width=2)

    # Block 4 (bottom-right): Park & Lake
    draw.ellipse([310, 310, 460, 460], fill=(40, 110, 160), outline=(30, 80, 120), width=3)
    draw.ellipse([340, 340, 430, 430], fill=(30, 90, 140))

    out_path = Path(__file__).resolve().parent.parent / "test_datasets" / "optical_mode" / "sample_aerial.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Created {out_path}")

if __name__ == "__main__":
    create_sample_scene()
