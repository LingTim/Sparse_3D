import os
import argparse
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from rembg import remove

# 1. 處理終端機傳入的指令參數
parser = argparse.ArgumentParser(description="Zero123++ Image to Multi-view Generator")
parser.add_argument("img_path", type=str, help="輸入圖片的檔案路徑")
args = parser.parse_args()

input_img_path = args.img_path

# 檢查檔案是否存在
if not os.path.exists(input_img_path):
    print(f"錯誤：找不到圖片檔案 {input_img_path}")
    exit(1)

# 2. 建立 output 資料夾
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 載入 Zero123++ 模型
print("正在載入 Zero123++ 模型...")
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16, trust_remote_code=True
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')

# 讀取輸入圖片
print(f"正在處理圖片: {input_img_path}")
cond = Image.open(input_img_path).convert("RGBA")

# 執行 Zero123++ 生成 (預設輸出是一張 960x640 的 2x3 網格大圖)
print("正在生成多視角圖片 (這需要一些時間)...")
result = pipeline(cond, num_inference_steps=75).images[0]

# 將原始大圖存下來當作參考 (可選)
grid_path = os.path.join(output_dir, "grid_output.png")
result.save(grid_path)
print(f"已儲存原始網格大圖至: {grid_path}")

# 3 & 4. 切割成 6 張 320x320 的圖片並進行去背
print("開始切割圖片與去背處理...")
width, height = result.size  # 預期為 960 (寬) x 640 (高)
crop_size = 320

view_idx = 0
# 透過雙迴圈掃描網格 (外層為 Y 軸行，內層為 X 軸列)
for y in range(0, height, crop_size):
    for x in range(0, width, crop_size):
        # 定義切割框 (左, 上, 右, 下)
        box = (x, y, x + crop_size, y + crop_size)
        sliced_img = result.crop(box)
        
        # 使用 rembg 進行去背，會回傳帶有透明通道 (Alpha=0) 的 RGBA 圖片
        no_bg_img = remove(sliced_img)
        
        # 儲存去背後的單一視角圖
        out_path = os.path.join(output_dir, f"view_{view_idx}.png")
        no_bg_img.save(out_path)
        print(f"  -> 成功儲存並去背: {out_path}")
        
        view_idx += 1

print("✅ 所有視角圖片處理完成！")