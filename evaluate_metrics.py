import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 嘗試匯入評估指標庫，若未安裝請先：pip install scikit-image lpips
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

# ==================== 配置區域 ====================
# 請根據你的資料夾結構修改以下路徑
TEST_JSON_PATH = r"dataset/flower/transforms_test.json" 
RENDERS_DIR = r"output_model/flower/test/ours_3000/renders"
GT_DIR = r"output_model/flower/test/ours_3000/gt"
# =================================================

def load_and_preprocess_image(img_path):
    """
    讀取圖片，並強制將背景處理為「純白色」的 RGB 影像
    """
    img = Image.open(img_path)
    
    # 如果圖片有 Alpha 通道 (RGBA)，將透明背景換成純白底
    if img.mode == 'RGBA':
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3]) # 使用 Alpha 遮罩
        img = white_bg
    else:
        img = img.convert('RGB')
        
    return img

def main():
    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(f"找不到測試 JSON 檔案: {TEST_JSON_PATH}")

    with open(TEST_JSON_PATH, 'r') as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not frames:
        print("❌ JSON 檔案中沒有找到任何 frame 資料。")
        return

    # 初始化 LPIPS 評估網路 (使用 VGG 骨幹，這是 NeRF/3DGS 論文的標準作法)
    print("📥 正在載入 LPIPS (VGG) 模型...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg')
    loss_fn_lpips.eval()

    # 用於轉換 LPIPS 所需的 Tensor 格式 [-1, 1]
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    all_psnr = []
    all_ssim = []
    all_lpips = []

    print(f"🚀 開始計算共 {len(frames)} 張純白背景影像的評估指標...")

    for idx, frame in enumerate(frames):
        # 取得對應的檔名 (例如 test_view_0)
        file_path = frame["file_path"]
        
        # 3DGS render 輸出的命名通常是 00000.png, 00001.png 依此類推
        render_name = f"{idx:05d}.png"
        gt_name = f"{idx:05d}.png" # 或者是 f"{file_path}.png"，取決於你 gt 資料夾內的命名。3DGS render.py 預設兩邊都是 00000.png
        
        render_img_path = os.path.join(RENDERS_DIR, render_name)
        gt_img_path = os.path.join(GT_DIR, gt_name)

        # 雙重檢查：如果 3DGS 的 gt 資料夾保留了原始名稱，則切換成原始名稱
        if not os.path.exists(gt_img_path):
            gt_img_path = os.path.join(GT_DIR, f"{file_path}.png")

        if not os.path.exists(render_img_path) or not os.path.exists(gt_img_path):
            print(f"⚠️ 警告：找不到對應圖片，跳過視角 {idx} (Render: {render_name})")
            continue

        # 讀取並強制將背景填白 (確保雙方背景一致)
        render_pil = load_and_preprocess_image(render_img_path)
        gt_pil = load_and_preprocess_image(gt_img_path)

        # 轉為 numpy array 用於 PSNR 和 SSIM (範圍 0~255)
        render_np = np.array(render_pil)
        gt_np = np.array(gt_pil)

        # 1. 計算 PSNR
        psnr_val = psnr_func(gt_np, render_np, data_range=255)
        
        # 2. 計算 SSIM (指定 channel_axis=2 處理彩色圖)
        ssim_val = ssim_func(gt_np, render_np, data_range=255, channel_axis=2)

        # 3. 計算 LPIPS
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

        print(f"[{idx+1}/{len(frames)}] {file_path} -> PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")

    # ==================== 總結與輸出 ====================
    if all_psnr:
        mean_psnr = np.mean(all_psnr)
        mean_ssim = np.mean(all_ssim)
        mean_lpips = np.mean(all_lpips)

        print("\n================== 🎉 評估結果總結 ==================")
        print(f"統計視角總數: {len(all_psnr)} 張")
        print(f"📊 平均 PSNR  : {mean_psnr:.4f} dB  (越高越好，通常 >25 很優秀)")
        print(f"📊 平均 SSIM  : {mean_ssim:.4f}     (越接近 1 越好)")
        print(f"📊 平均 LPIPS : {mean_lpips:.4f}     (越低越好，越接近 0 代表人類視覺感知越一致)")
        print("====================================================")
        
        # 自動將數據存成一個 json 報告
        report_path = os.path.join(os.path.dirname(TEST_JSON_PATH), "metrics_report.json")
        report = {
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "num_images": len(all_psnr)
        }
        with open(report_path, "w") as rf:
            json.dump(report, rf, indent=4)
        print(f"報告已成功導出至: {report_path}")
    else:
        print("❌ 未成功計算任何指標，請檢查圖片路徑是否正確。")

if __name__ == "__main__":
    main()