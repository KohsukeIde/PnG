# visualization.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_reconstruction(points_3d, img1, img2, pts1_inliers, pts2_inliers):
    """再構成結果を可視化します。

    Args:
        points_3d: np.ndarray[N, 3], 再構成された3D点群
        img1: np.ndarray, 画像1
        img2: np.ndarray, 画像2
        pts1_inliers: np.ndarray[N, 2], 画像1のインライア対応点
        pts2_inliers: np.ndarray[N, 2], 画像2のインライア対応点
    """
    # 出力ディレクトリの作成
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # インライア点が存在するか確認
    if pts1_inliers.size == 0 or pts2_inliers.size == 0:
        print("可視化するインライア点がありません。")
        return

    plt.figure(figsize=(15, 5))

    # 画像1にインライア点をプロット
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.scatter(pts1_inliers[:, 0], pts1_inliers[:, 1], c='r', marker='o')
    plt.title('Image 1 with Inlier Points')

    # 画像2にインライア点をプロット
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2_inliers[:, 0], pts2_inliers[:, 1], c='r', marker='o')
    plt.title('Image 2 with Inlier Points')

    plt.savefig(os.path.join(output_dir, 'inlier_points.png'))
    plt.close()

    # 3D点群の可視化
    if points_3d.size == 0:
        print("可視化する3D点がありません。")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Reconstructed 3D Points')
    plt.savefig(os.path.join(output_dir, '3d_points.png'))
    plt.close()

def save_points_to_ply(points_3d, filename='reconstructed_points.ply', colors=None):
    """3D点群をPLYファイルに保存します。

    Args:
        points_3d: np.ndarray[N, 3], 3D点群
        filename: str, 出力PLYファイルの名前
        colors: np.ndarray[N, 3], 各点のRGB値（0-255の範囲）
    """
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    num_points = points_3d.shape[0]
    header = f'''ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
'''
    if colors is not None:
        header += '''property uchar red
property uchar green
property uchar blue
'''

    header += 'end_header\n'

    with open(file_path, 'w') as f:
        f.write(header)
        if colors is not None:
            for point, color in zip(points_3d, colors):
                # カラーデータを0-255の範囲に変換
                color = np.clip(color, 0, 255).astype(int)
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        else:
            for point in points_3d:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
