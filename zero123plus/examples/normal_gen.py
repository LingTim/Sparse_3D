import os
import cv2
import copy
import numpy
import torch
import argparse
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel
from matting_postprocess import postprocess

# 1. 處理終端機傳入的指令參數
parser = argparse.ArgumentParser(description="Zero123++ Image to Multi-view Color & Normal Generator")
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


def rescale(single_res, input_image, ratio=0.95):
    # Rescale and recenter
    image_arr = numpy.array(input_image)
    ret, mask = cv2.threshold(numpy.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio)
    padded_image = numpy.zeros((side_len, side_len, 4), dtype=numpy.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
    return rgba


# 3. 載入 Zero123++ 與 ControlNet 法線模型
print("正在載入 Zero123++ 與 ControlNet 法線模型...")
pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16, local_files_only=False, trust_remote_code=True
)
normal_pipeline = copy.copy(pipeline)
normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
    "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16, local_files_only=False
), conditioning_scale=1.0)

pipeline.to("cuda:0", torch.float16)
normal_pipeline.to("cuda:0", torch.float16)

# 4. 讀取並處理輸入圖片
print(f"正在處理圖片: {input_img_path}")
cond = Image.open(input_img_path).convert("RGBA")

# 如果物體在圖片中佔比太小，可自由開啟下行進行重新縮放對中
# cond = rescale(512, cond)

# 5. 執行 Pipeline 生成大圖
print("正在生成多視角顏色網格圖...")
genimg = pipeline(
    cond,
    prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
).images[0]

print("正在生成多視角法線網格圖...")
normalimg = normal_pipeline(
    cond, depth_image=genimg,
    prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
).images[0]

# 6. 幾何去背後處理
print("正在執行高級幾何去背與優化...")
genimg, normalimg = postprocess(genimg, normalimg)

# ✨ 核心技術優化：將顏色圖的 Alpha 遮罩完美注入法線圖中，實現法線圖透明去背
_, _, _, alpha_channel = genimg.split()
normalimg_rgba = normalimg.convert("RGB")
normalimg_rgba.putalpha(alpha_channel)

# 儲存網格大圖
grid_colors_path = os.path.join(output_dir, "grid_colors.png")
grid_normals_path = os.path.join(output_dir, "grid_normals.png")
genimg.save(grid_colors_path)
normalimg_rgba.save(grid_normals_path)
print(f"已儲存顏色網格大圖至: {grid_colors_path}")
print(f"已儲存法線網格大圖至: {grid_normals_path}")

# 7. 雙迴圈掃描並切割成 6 張 320x320 的獨立視角圖
print("開始切割 6 視角的顏色圖與法線圖...")
width, height = genimg.size  # 預期為 640 (寬) x 960 (高)
crop_size = 320

view_idx = 0
for y in range(0, height, crop_size):
    for x in range(0, width, crop_size):
        box = (x, y, x + crop_size, y + crop_size)
        
        # 切割並儲存顏色圖 (RGBA)
        sliced_color = genimg.crop(box)
        out_color_path = os.path.join(output_dir, f"view_{view_idx}.png")
        sliced_color.save(out_color_path)
        
        # 切割並儲存法線圖 (RGBA)
        sliced_normal = normalimg_rgba.crop(box)
        out_normal_path = os.path.join(output_dir, f"normal_{view_idx}.png")
        sliced_normal.save(out_normal_path)
        
        print(f"  -> 成功儲存第 {view_idx} 視角: {out_color_path} 與 {out_normal_path}")
        view_idx += 1

print("✅ 所有顏色與法線視角圖片處理完成！")