import os
import argparse
from PIL import Image

# 1. 設置參數解析
parser = argparse.ArgumentParser(description="Split matted 640x960 grid images into 6 separate 320x320 views")
parser.add_argument("--color_grid", type=str, default="output/grid_colors_matted.png", help="去背後的顏色網格大圖路徑")
parser.add_argument("--normal_grid", type=str, default="output/grid_normals_matted.png", help="去背後的法線網格大圖路徑")
parser.add_argument("--out_dir", type=str, default="output", help="分割後圖片要儲存的資料夾")
args = parser.parse_args()

# 建立輸出資料夾
os.makedirs(args.out_dir, exist_ok=True)
crop_size = 320

# 2. 處理並分割顏色大圖
if os.path.exists(args.color_grid):
    print(f"正在分割顏色網格圖: {args.color_grid}")
    img = Image.open(args.color_grid).convert("RGBA")  # 確保保留去背的 Alpha 透明通道
    width, height = img.size
    
    view_idx = 0
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            box = (x, y, x + crop_size, y + crop_size)
            sliced = img.crop(box)
            out_path = os.path.join(args.out_dir, f"view_{view_idx}.png")
            sliced.save(out_path)
            print(f"  -> 儲存顏色視角: {out_path}")
            view_idx += 1
else:
    print(f"提示：找不到顏色網格圖 {args.color_grid}，跳過顏色分割。")

# 3. 處理並分割法線大圖
if os.path.exists(args.normal_grid):
    print(f"正在分割法線網格圖: {args.normal_grid}")
    img = Image.open(args.normal_grid).convert("RGBA")  # 確保保留去背的 Alpha 透明通道
    width, height = img.size
    
    view_idx = 0
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            box = (x, y, x + crop_size, y + crop_size)
            sliced = img.crop(box)
            out_path = os.path.join(args.out_dir, f"normal_{view_idx}.png")
            sliced.save(out_path)
            print(f"  -> 儲存法線視角: {out_path}")
            view_idx += 1
else:
    print(f"提示：找不到法線網格圖 {args.normal_grid}，跳過法線分割。")

print("✅ 所有視角圖片分割完成！")