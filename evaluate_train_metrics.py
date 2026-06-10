import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("❌ 錯誤：請先安裝 skimage。執行：pip install scikit-image")
    exit(1)

try:
    import lpips
except ImportError:
    print("❌ 錯誤：請先安裝 lpips。執行：pip install lpips")
    exit(1)

# ==================== 🛠️ 訓練集配置區域 ====================
# 1. 設定你的 3DGS 訓練集渲染輸出主資料夾 (請根據你實際的資料夾名稱改為 ours_3000 或 our3000)
BASE_DIR = os.path.join("output_model", "tree1", "train", "ours_3000") 

RENDERS_DIR = os.path.join(BASE_DIR, "renders")
GT_DIR = os.path.join(BASE_DIR, "gt")

# 2. 自動尋找 JSON 檔案的路徑
# 優先檢查 train 資料夾下有沒有 JSON，如果沒有，就去原始 dataset 找 transforms_train.json
JSON_PATH = os.path.join(BASE_DIR, "transforms_train.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join("dataset", "carrot", "transforms_train.json")
# =========================================================

def load_and_preprocess_image(img_path):
    """
    讀取圖片，並強制將背景處理為「純白色」的 RGB 影像
    """
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        img = white_bg
    else:
        img = img.convert('RGB')
    return img

def main():
    print(f"🔍 正在檢查訓練集 JSON 檔案: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"❌ 找不到訓練集的 JSON 檔案，請檢查路徑！\n目前嘗試路徑: {JSON_PATH}")

    with open(JSON_PATH, 'r') as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not frames:
        print("❌ JSON 檔案中沒有找到任何 frame 資料。")
        return

    print("📥 正在載入 LPIPS (VGG) 模型...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg')
    loss_fn_lpips.eval()

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    all_psnr = []
    all_ssim = []
    all_lpips = []

    print(f"🚀 開始計算共 {len(frames)} 張【訓練集】影像的優化指標...")

    for idx, frame in enumerate(frames):
        file_path = frame["file_path"]
        
        # 3DGS 渲染出來的命名通常是 00000.png, 00001.png...
        render_name = f"{idx:05d}.png"
        gt_name = f"{idx:05d}.png"
        
        render_img_path = os.path.join(RENDERS_DIR, render_name)
        gt_img_path = os.path.join(GT_DIR, gt_name)

        # 備用檢查：如果資料夾保留了原始名稱 (如 view_0.png)
        if not os.path.exists(gt_img_path):
            gt_img_path = os.path.join(GT_DIR, f"{file_path}.png")

        if not os.path.exists(render_img_path) or not os.path.exists(gt_img_path):
            print(f"⚠️ 警告：找不到圖片，跳過訓練視角 {idx} (預期檔名: {render_name})")
            continue

        render_pil = load_and_preprocess_image(render_img_path)
        gt_pil = load_and_preprocess_image(gt_img_path)

        render_np = np.array(render_pil)
        gt_np = np.array(gt_pil)

        # 1. PSNR
        psnr_val = psnr_func(gt_np, render_np, data_range=255)
        
        # 2. SSIM
        ssim_val = ssim_func(gt_np, render_np, data_range=255, channel_axis=2)

        # 3. LPIPS
        with torch.no_grad():
            render_t = transform_tensor(render_pil).unsqueeze(0)
            gt_t = transform_tensor(gt_pil).unsqueeze(0)
            if torch.cuda.is_available():
                render_t = render_t.cuda()
                gt_t = gt_t.cuda()
            lpips_val = loss_fn_lpips(render_t, gt_t).item()

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_lpips.append(lpips_val)

        print(f"[{idx+1}/{len(frames)}] {file_path} -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")

    if all_psnr:
        mean_psnr = np.mean(all_psnr)
        mean_ssim = np.mean(all_ssim)
        mean_lpips = np.mean(all_lpips)

        print("\n================== 🎉 訓練集 (Train) 評估結果 ==================")
        print(f"統計視角總數: {len(all_psnr)} 張 ( should be 6 views )")
        print(f"📊 平均 PSNR  : {mean_psnr:.4f} dB")
        print(f"📊 平均 SSIM  : {mean_ssim:.4f}")
        print(f"📊 平均 LPIPS : {mean_lpips:.4f}")
        print("===============================================================")
        
        # 儲存報告
        report_path = os.path.join(BASE_DIR, "train_metrics_report.json")
        report = {
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "num_images": len(all_psnr)
        }
        with open(report_path, "w") as rf:
            json.dump(report, rf, indent=4)
        print(f"報告已存至: {report_path}")
    else:
        print("❌ 未成功計算任何指標，請檢查 renders 與 gt 資料夾內是否有圖片。")

if __name__ == "__main__":
    main()