import os
import json
import numpy as np
from PIL import Image

def main():
    output_dir = "dataset/flower"
    json_path = os.path.join(output_dir, "transforms_train.json")
    out_ply_path = os.path.join(output_dir, "points3d.ply")
    
    if not os.path.exists(json_path):
        print(f"錯誤：找不到相機矩陣檔案 {json_path}，請先執行 generate_cameras.py")
        return

    with open(json_path, "r") as f:
        meta = json.load(f)

    W, H = 320, 320
    cx, cy = 160, 160
    camera_angle_x = meta["camera_angle_x"]
    focal = cx / np.tan(camera_angle_x / 2.0)

    frames_data = []
    for frame in meta["frames"]:
        view_name = frame["file_path"] # 例如 "view_0"
        img_path = os.path.join(output_dir, f"{view_name}.png")
        
        # ✨ 新增：自動對應讀取你去背後的獨立法線圖 (例如 "normal_0.png")
        norm_name = view_name.replace("view_", "normal_")
        norm_path = os.path.join(output_dir, f"{norm_name}.png")
        
        if not os.path.exists(img_path):
            print(f"警告：找不到顏色圖片 {img_path}，跳過此視角")
            continue
            
        # 讀取顏色與遮罩
        img = Image.open(img_path).convert("RGBA")
        img_arr = np.array(img)
        rgb = img_arr[:, :, :3]
        alpha = img_arr[:, :, 3]

        # ✨ 新增：讀取並解碼法線圖
        if os.path.exists(norm_path):
            norm_img = Image.open(norm_path).convert("RGB")
            norm_arr = np.array(norm_img).astype(np.float32) / 255.0
            # 將 0~1 的 RGB 換算回 -1~1 的相機空間法線向量 (NX, NY, NZ)
            norm_cam = norm_arr * 2.0 - 1.0
            
            # 💡 依據 Zero123++ / ControlNet 標準常規常需轉換 OpenGL -> OpenCV 座標系
            # 如果後續看點雲發現法線朝內，可以將下面兩行解開註解來修正方向：
            # norm_cam[:, :, 1] *= -1  # 翻轉 Y
            # norm_cam[:, :, 2] *= -1  # 翻轉 Z
        else:
            print(f"警告：找不到對應的法線圖 {norm_path}，此視角法線將填空")
            norm_cam = np.zeros((H, W, 3), dtype=np.float32)

        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV 座標轉換
        
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        frames_data.append({
            "R": R, "T": T, "rgb": rgb, "alpha": alpha, "norm_cam": norm_cam
        })

    if len(frames_data) == 0:
        print("錯誤：沒有載入任何有效的視角圖片")
        return

    print(f"成功載入 {len(frames_data)} 個視角的相機、去背遮罩與法線圖。開始空間雕刻與法線融合...")

    resolution = 128  
    grid_range = np.linspace(-0.5, 0.5, resolution)
    X, Y, Z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    pts_world = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

    N = pts_world.shape[0]
    keep_mask = np.ones(N, dtype=bool)
    accum_colors = np.zeros((N, 3), dtype=np.float32)
    
    # ✨ 新增：用來累積 3D 世界座標系法線的矩陣
    accum_normals = np.zeros((N, 3), dtype=np.float32)
    view_counts = np.zeros(N, dtype=np.float32)

    for frame in frames_data:
        R, T = frame["R"], frame["T"]
        alpha, rgb, norm_cam = frame["alpha"], frame["rgb"], frame["norm_cam"]

        pts_cam = (R @ pts_world.T).T + T
        x_c, y_c, z_c = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

        z_c = np.where(z_c == 0, 1e-5, z_c)
        in_front = z_c > 0

        u = focal * (x_c / z_c) + cx
        v = focal * (y_c / z_c) + cy

        u_idx = np.round(u).astype(int)
        v_idx = np.round(v).astype(int)

        in_bounds = in_front & (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)

        current_view_keep = np.zeros(N, dtype=bool)
        if np.any(in_bounds):
            valid_u = u_idx[in_bounds]
            valid_v = v_idx[in_bounds]
            is_foreground = alpha[valid_v, valid_u] > 50
            current_view_keep[in_bounds] = is_foreground
            
            # 累積顏色
            accum_colors[in_bounds] = np.where(
                is_foreground[:, None], 
                accum_colors[in_bounds] + rgb[valid_v, valid_u], 
                accum_colors[in_bounds]
            )
            
            # ✨ 核心修正：將該像素的「相機空間法線」乘上旋轉矩陣的逆矩陣（即 R 的轉置 R.T），轉回「世界空間法線」
            pix_norm_cam = norm_cam[valid_v, valid_u] # (符合條件的點數, 3)
            pix_norm_world = pix_norm_cam @ R         # 矩陣旋轉回世界座標系
            
            # 累積法線
            accum_normals[in_bounds] = np.where(
                is_foreground[:, None],
                accum_normals[in_bounds] + pix_norm_world,
                accum_normals[in_bounds]
            )
            
            view_counts[in_bounds] += is_foreground

        keep_mask = keep_mask & current_view_keep

    final_pts = pts_world[keep_mask]
    final_counts = view_counts[keep_mask]
    final_counts = np.where(final_counts == 0, 1, final_counts)
    
    final_clrs = accum_colors[keep_mask] / final_counts[:, None]
    
    # ✨ 新增：計算平均法線，並進行單位化 (Normalize) 確保向量長度為 1
    final_norms = accum_normals[keep_mask] / final_counts[:, None]
    norm_lens = np.linalg.norm(final_norms, axis=-1, keepdims=True)
    norm_lens = np.where(norm_lens == 0, 1e-5, norm_lens)
    final_norms = final_norms / norm_lens

    print(f"雕刻完成！精煉出 {len(final_pts)} 個物體表面點，並已融合真實世界表面法線。")

    print(f"正在將包含法線向量的點雲儲存至: {out_ply_path}")
    with open(out_ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(final_pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        
        # ✨ 新增：將真正的 final_norms (pt_n) 寫入檔案中，取代原本的 0.0 0.0 0.0
        for pt, pt_n, clr in zip(final_pts, final_norms, final_clrs):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {pt_n[0]:.4f} {pt_n[1]:.4f} {pt_n[2]:.4f} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")

    print("✅ points3d.ply 點雲（含真實法線引導）生成成功！")

if __name__ == "__main__":
    main()