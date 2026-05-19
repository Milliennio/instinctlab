import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(data_dir: str, num_samples: int = 5):
    """加载并可视化收集的深度图和真实高程图数据"""
    
    # 查找所有的 .npz 文件
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print(f"[Error] No .npz files found in {data_dir}")
        return

    # 为了简单起见，我们读取找到的第一个 Chunk
    data_path = npz_files[0]
    print(f"[INFO] Loading data from {data_path}...")
    data = np.load(data_path)
    
    depth_history = data['depth_history']  # Shape: (N, 8, 18, 32)
    gt_elevation = data['gt_elevation']    # Shape: (N, 31, 21)
    
    N = depth_history.shape[0]
    print(f"[INFO] Total valid samples in this chunk: {N}")
    
    # 随机抽取需要可视化的样本索引
    indices = np.random.choice(N, min(num_samples, N), replace=False)
    
    for idx in indices:
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f"Sample Index: {idx}", fontsize=16)
        
        # 1. 绘制 8 帧历史深度图 (第一排)
        # 假设深度图已经被截断并归一化到了 [0, 1]
        for i in range(8):
            ax = fig.add_subplot(2, 8, i + 1)
            im = ax.imshow(depth_history[idx, i], cmap='viridis', vmin=0, vmax=1.0)
            # i=0是最旧的帧，i=7是最新的帧
            ax.set_title(f"Depth T-{7-i}")
            ax.axis('off')
            
        # 2. 绘制当前脚下盲区的真实高程图 GT (第二排居中)
        ax_gt = fig.add_subplot(2, 1, 2)
        # 使用 'terrain' colormap 以更好地显示地形高低起伏
        # extent 帮助映射物理尺寸：左(0.2) 右(-0.2)，下(-0.3) 上(0.3)
        im_gt = ax_gt.imshow(gt_elevation[idx], cmap='terrain', extent=[0.2, -0.2, -0.3, 0.3])
        ax_gt.set_title("Ground Truth Elevation Map (Blind Spot 0.6m x 0.4m)")
        ax_gt.set_ylabel("Forward / Backward (X) [m]")
        ax_gt.set_xlabel("Left / Right (Y) [m]")
        # 移除 ax_gt.axis('off') 以保留坐标轴刻度，方便观察物理尺度
        
        # 添加 Colorbar 以查看具体的高程数值 (单位: 米)
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, label="Relative Height (m)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize collected depth dataset.")
    parser.add_argument("--data_dir", type=str, default="data_collection", help="Directory where .npz files are saved.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    args = parser.parse_args()
    
    visualize_dataset(args.data_dir, args.num_samples)