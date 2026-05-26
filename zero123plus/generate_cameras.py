import os
import json
import numpy as np

def get_c2w_matrix(azimuth_deg, elevation_deg, radius=1.5):
    # 將角度轉換為弧度
    azi = np.deg2rad(azimuth_deg)
    ele = np.deg2rad(elevation_deg)
    
    # 1. 根據球座標系計算相機在 3D 空間中的位置 (X, Y, Z)
    # 假設 Y 軸朝上 (OpenGL 座標系慣例)
    x = radius * np.cos(ele) * np.sin(azi)
    y = radius * np.sin(ele)
    z = radius * np.cos(ele) * np.cos(azi)
    camera_pos = np.array([x, y, z])
    
    # 2. 計算 LookAt 矩陣 (相機永遠看向原點 [0, 0, 0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    
    # 計算相機座標系的 Z, X, Y 軸向量
    z_axis = camera_pos - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 3. 組合 4x4 的 Camera-to-World (c2w) 矩陣
    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_pos
    
    return c2w

def main():
    # v1.2 版本的固定參數
    camera_radius = 1.5
    fov_deg = 30.0
    camera_angle_x = np.deg2rad(fov_deg)
    
    # 6 個視角的參數 (對應 2 欄 3 列的切割順序)
    views_params = [
        {"id": 0, "azi": 30,  "ele": 20},
        {"id": 1, "azi": 90,  "ele": -10},
        {"id": 2, "azi": 150, "ele": 20},
        {"id": 3, "azi": 210, "ele": -10},
        {"id": 4, "azi": 270, "ele": 20},
        {"id": 5, "azi": 330, "ele": -10},
    ]
    
    frames = []
    for params in views_params:
        c2w = get_c2w_matrix(params["azi"], params["ele"], radius=camera_radius)
        
        # 建立單個 frame 的資訊
        frame = {
            "file_path": f"view_{params['id']}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
    
    # 組合最終的 JSON 結構
    transforms_dict = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    # 輸出至 JSON 檔案
    output_path = "zero123plus/output/transforms_train.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transforms_dict, f, indent=4)
        
    print(f"✅ 成功產生相機參數檔案: {output_path}")

if __name__ == "__main__":
    main()