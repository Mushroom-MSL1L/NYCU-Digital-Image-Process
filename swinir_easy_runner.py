#!/usr/bin/env python3
"""
SwinIR Image Denoising 一鍵執行腳本
適用於期末專題 - Image Denoising
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import shutil

def setup_directories():
    """創建必要的目錄結構"""
    dirs = [
        "model_zoo/swinir",
        "testsets/project_images", 
        "results"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 創建目錄: {dir_path}")

def download_models():
    """下載SwinIR預訓練模型"""
    models = {
        "noise15": "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
        "noise25": "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth", 
        "noise50": "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    }
    
    base_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    
    for name, filename in models.items():
        model_path = f"model_zoo/swinir/{filename}"
        
        if os.path.exists(model_path):
            print(f"✓ 模型已存在: {filename}")
            continue
            
        print(f"📥 下載 {filename}...")
        try:
            url = base_url + filename
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ 下載完成: {filename}")
            
        except Exception as e:
            print(f"✗ 下載失敗 {filename}: {e}")
            return False
    
    return True

def prepare_test_images():
    """準備測試圖像"""
    test_dir = Path("testsets/project_images")
    
    if not any(test_dir.iterdir()):
        print("📁 請將你的有噪聲圖像放入以下目錄:")
        print(f"   {test_dir.absolute()}")
        print("\n支援格式: .jpg, .jpeg, .png, .bmp")
        
        # 等待用戶放入圖像
        input("\n按 Enter 鍵繼續（確保已放入測試圖像）...")
    
    # 檢查圖像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(test_dir.glob(f"*{ext}")))
        images.extend(list(test_dir.glob(f"*{ext.upper()}")))
    
    print(f"✓ 找到 {len(images)} 個圖像文件")
    for img in images[:5]:  # 顯示前5個
        print(f"  - {img.name}")
    
    if len(images) > 5:
        print(f"  ... 還有 {len(images) - 5} 個文件")
    
    return len(images) > 0

def run_denoising(noise_levels=[25], use_tile=False):
    """執行圖像降噪"""
    results_summary = {}
    
    for noise in noise_levels:
        print(f"\n🚀 執行降噪 - 噪聲級別: {noise}")
        
        # 構建命令
        cmd = [
            "python", "main_test_swinir.py",
            "--task", "color_dn",
            "--noise", str(noise),
            "--model_path", f"model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth",
            "--folder_gt", "testsets/project_images"
        ]
        
        # 如果使用tile模式（適合大圖像或記憶體不足）
        if use_tile:
            cmd.extend(["--tile", "400", "--tile_overlap", "32"])
        
        try:
            # 執行命令
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 重命名結果資料夾
            old_dir = f"results/swinir_color_dn_noise{noise}"
            new_dir = f"results/project_denoised_noise{noise}"
            
            if os.path.exists(old_dir):
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir)
                shutil.move(old_dir, new_dir)
                
                print(f"✓ 降噪完成 - 結果保存在: {new_dir}")
                results_summary[noise] = new_dir
            else:
                print(f"⚠️  結果目錄未找到: {old_dir}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ 執行失敗 - 噪聲級別 {noise}")
            print(f"錯誤信息: {e.stderr}")
            
            # 如果記憶體不足，建議使用tile模式
            if "out of memory" in e.stderr.lower() or "cuda" in e.stderr.lower():
                print("💡 建議使用tile模式重試")
    
    return results_summary

def compare_results(results_summary):
    """比較不同噪聲級別的結果"""
    if not results_summary:
        return
    
    print("\n📊 結果比較:")
    print("=" * 50)
    
    for noise, result_dir in results_summary.items():
        if os.path.exists(result_dir):
            file_count = len(list(Path(result_dir).glob("*.png")))
            print(f"噪聲級別 {noise:2d}: {file_count} 個處理結果 -> {result_dir}")
    
    print("\n💡 建議:")
    print("1. 查看不同噪聲級別的結果，選擇效果最好的")
    print("2. 可以將多個結果組合成ensemble方法")
    print("3. 對於期末專題，建議展示多個noise level的比較")

def create_ensemble_script():
    """創建ensemble方法的腳本"""
    ensemble_code = '''
import cv2
import numpy as np
from pathlib import Path

def ensemble_denoise(image_paths, output_path, weights=None):
    """
    集成多個降噪結果
    image_paths: 不同noise level的結果圖像路徑列表
    weights: 權重列表，如果為None則平均
    """
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img.astype(np.float32))
    
    if not images:
        return False
    
    if weights is None:
        weights = [1.0 / len(images)] * len(images)
    
    # 加權平均
    result = np.zeros_like(images[0])
    for img, weight in zip(images, weights):
        result += img * weight
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, result)
    return True

# 使用範例
if __name__ == "__main__":
    # 對每個原始圖像進行ensemble
    for noise15_img in Path("results/project_denoised_noise15").glob("*.png"):
        base_name = noise15_img.stem.replace("_SwinIR", "")
        
        # 找到對應的其他noise level結果
        noise25_img = Path("results/project_denoised_noise25") / f"{base_name}_SwinIR.png"
        noise50_img = Path("results/project_denoised_noise50") / f"{base_name}_SwinIR.png"
        
        if noise25_img.exists() and noise50_img.exists():
            # 給予不同權重（可以調整）
            ensemble_denoise(
                [noise15_img, noise25_img, noise50_img],
                f"results/ensemble_{base_name}.png",
                weights=[0.4, 0.4, 0.2]  # 偏重輕、中等噪聲的結果
            )
            print(f"Ensemble completed: {base_name}")
'''
    
    with open("create_ensemble.py", "w") as f:
        f.write(ensemble_code)
    
    print("✓ 已創建ensemble腳本: create_ensemble.py")

def main():
    """主函數"""
    print("🎯 SwinIR Image Denoising - 期末專題助手")
    print("=" * 60)
    
    # 步驟1: 設置目錄
    print("\n📁 設置目錄結構...")
    setup_directories()
    
    # 步驟2: 下載模型
    print("\n📥 下載預訓練模型...")
    if not download_models():
        print("❌ 模型下載失敗，請檢查網路連接")
        return
    
    # 步驟3: 準備測試圖像
    print("\n🖼️  準備測試圖像...")
    if not prepare_test_images():
        print("❌ 沒有找到測試圖像")
        return
    
    # 步驟4: 選擇降噪參數
    print("\n⚙️  選擇降噪參數:")
    print("1. 僅測試中等噪聲 (noise=25)")
    print("2. 測試所有噪聲級別 (15, 25, 50)")
    print("3. 自定義")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == "1":
        noise_levels = [25]
    elif choice == "2":
        noise_levels = [15, 25, 50]
    elif choice == "3":
        custom = input("輸入噪聲級別 (用空格分隔，如: 15 25): ")
        noise_levels = [int(x) for x in custom.split()]
    else:
        noise_levels = [25]  # 默認
    
    # 詢問是否使用tile模式
    use_tile = input("\n是否使用tile模式? (推薦用於大圖像或Mac) [y/N]: ").lower() == 'y'
    
    # 步驟5: 執行降噪
    print(f"\n🚀 開始降噪處理 - 噪聲級別: {noise_levels}")
    results = run_denoising(noise_levels, use_tile)
    
    # 步驟6: 結果比較
    compare_results(results)
    
    # 步驟7: 創建ensemble腳本
    if len(results) > 1:
        create_ensemble = input("\n是否創建ensemble方法腳本? [y/N]: ").lower() == 'y'
        if create_ensemble:
            create_ensemble_script()
    
    print("\n🎉 處理完成！")
    print("\n📝 期末專題建議:")
    print("1. 比較不同noise level的PSNR/SSIM結果")
    print("2. 展示視覺效果對比 (原圖 vs 降噪後)")
    print("3. 如果有多個結果，可以實作ensemble方法")
    print("4. 分析各種方法的優缺點")

if __name__ == "__main__":
    main()