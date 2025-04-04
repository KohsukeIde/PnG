import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import argparse
import torch
import pickle
import cv2
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

from src.reconstructor.view_selector import ViewSelector

def parse_args():
    parser = argparse.ArgumentParser(description="Initial Pair Selection Test")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/textureless/images",
        help="Path to the data directory containing images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="initial_pair_test_results",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--min_overlap", 
        type=float, 
        default=0.2,
        help="Minimum overlap ratio for pair selection"
    )
    parser.add_argument(
        "--max_overlap", 
        type=float, 
        default=0.6,
        help="Maximum overlap ratio for pair selection"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=0,
        help="Number of images to sample for tests (default: 20, use 0 for all images)"
    )
    parser.add_argument(
        "--force_clip", 
        action="store_true",
        help="Force using CLIP even if it fails to initialize"
    )
    return parser.parse_args()

def get_image_paths(image_dir, sample_size=0):
    """画像ディレクトリから画像パスを取得し、必要に応じてサンプリング"""
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    image_files.sort()  # 一貫性のためにソート
    
    if sample_size > 0 and sample_size < len(image_files):
        # 均等に分散するようにサンプリング
        indices = np.linspace(0, len(image_files)-1, sample_size, dtype=int)
        return [image_files[i] for i in indices]
    
    return image_files

def compute_similarity_matrices(image_paths, min_overlap, max_overlap, output_dir):
    """CLIPとSIFTの両方で類似度行列を計算し保存"""
    # CLIP Selector
    clip_selector = ViewSelector(
        image_dir=os.path.dirname(image_paths[0]),
        feature_type='clip',
        min_overlap_ratio=min_overlap,
        max_overlap_ratio=max_overlap,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # SIFT Selector
    sift_selector = ViewSelector(
        image_dir=os.path.dirname(image_paths[0]),
        feature_type='sift',
        vocab_size=200,
        min_overlap_ratio=min_overlap,
        max_overlap_ratio=max_overlap
    )
    
    # Process images with both methods
    print("Processing images with CLIP...")
    clip_selector.process_images(image_paths)
    
    print("Processing images with SIFT...")
    sift_selector.process_images(image_paths)
    
    # Compute similarity matrices
    n = len(image_paths)
    clip_similarity = np.zeros((n, n))
    sift_similarity = np.zeros((n, n))
    
    print("Computing similarity matrices...")
    
    # CLIP Similarity
    if clip_selector.feature_type == 'clip' and clip_selector.clip_model is not None:
        for i, img1 in enumerate(tqdm(image_paths, desc="CLIP Similarity")):
            if img1 not in clip_selector.clip_features:
                continue
                
            for j, img2 in enumerate(image_paths):
                if i <= j:  # 対称行列なので半分だけ計算
                    if img2 not in clip_selector.clip_features:
                        continue
                        
                    if i == j:
                        clip_similarity[i, j] = 1.0
                    else:
                        feat1 = clip_selector.clip_features[img1]
                        feat2 = clip_selector.clip_features[img2]
                        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                        clip_similarity[i, j] = sim
                        clip_similarity[j, i] = sim
    else:
        print("Warning: CLIP not available, using random values")
        clip_similarity = np.random.rand(n, n)
        clip_similarity = (clip_similarity + clip_similarity.T) / 2  # 対称化
        np.fill_diagonal(clip_similarity, 1.0)

    # SIFT Similarity
    if sift_selector.codebook is not None:
        for i, img1 in enumerate(tqdm(image_paths, desc="SIFT Similarity")):
            if img1 not in sift_selector.image_histograms:
                continue
                
            for j, img2 in enumerate(image_paths):
                if i <= j:
                    if img2 not in sift_selector.image_histograms:
                        continue
                        
                    if i == j:
                        sift_similarity[i, j] = 1.0
                    else:
                        hist1 = sift_selector.image_histograms[img1]
                        hist2 = sift_selector.image_histograms[img2]
                        # ヒストグラム交差法（Histogram Intersection）
                        sim = np.sum(np.minimum(hist1, hist2))
                        sift_similarity[i, j] = sim
                        sift_similarity[j, i] = sim
    else:
        print("Warning: SIFT codebook not available, using random values")
        sift_similarity = np.random.rand(n, n)
        sift_similarity = (sift_similarity + sift_similarity.T) / 2  # 対称化
        np.fill_diagonal(sift_similarity, 1.0)
    
    # 類似度行列を可視化して保存
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#0000FF', '#FFFFFF', '#FF0000'], N=256)
    
    im1 = ax1.imshow(clip_similarity, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('CLIP Similarity Matrix')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(sift_similarity, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title('SIFT Similarity Matrix')
    plt.colorbar(im2, ax=ax2)
    
    # 画像名が多すぎる場合は間引く
    if n <= 20:
        for ax in [ax1, ax2]:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([os.path.basename(f) for f in image_paths], rotation=90)
            ax.set_yticklabels([os.path.basename(f) for f in image_paths])
    else:
        # 間引いて表示
        step = max(1, n // 10)
        indices = range(0, n, step)
        for ax in [ax1, ax2]:
            ax.set_xticks(indices)
            ax.set_yticks(indices)
            ax.set_xticklabels([os.path.basename(image_paths[i]) for i in indices], rotation=90)
            ax.set_yticklabels([os.path.basename(image_paths[i]) for i in indices])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_matrices.png"))
    plt.close()
    
    # 類似度分布のヒストグラムを作成
    plt.figure(figsize=(12, 6))
    
    # 上三角行列から値を抽出（対角要素を除く）
    clip_values = clip_similarity[np.triu_indices(n, k=1)]
    sift_values = sift_similarity[np.triu_indices(n, k=1)]
    
    plt.hist([clip_values, sift_values], bins=30, label=['CLIP', 'SIFT'], alpha=0.7)
    plt.xlabel('Similarity Value')
    plt.ylabel('Frequency')
    plt.title('Similarity Value Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "similarity_histogram.png"))
    plt.close()
    
    return clip_selector, sift_selector, clip_similarity, sift_similarity

def select_best_pairs(similarity_matrix, image_paths, min_overlap, max_overlap, top_k=5):
    """類似度行列から最良のペアをtop_k個選択"""
    n = len(image_paths)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = similarity_matrix[i, j]
            
            # 指定範囲内の類似度を持つペアのみ考慮
            if min_overlap <= similarity <= max_overlap:
                pairs.append((i, j, similarity))
    
    # 類似度でソート（高い順）
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 上位k個を返す（ただし条件を満たすペアが足りない場合は全て返す）
    return pairs[:min(top_k, len(pairs))]

def visualize_selected_pairs(image_paths, best_pairs, method, output_dir):
    """選択されたペアを可視化"""
    for idx, (i, j, sim) in enumerate(best_pairs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        img1 = Image.open(image_paths[i])
        img2 = Image.open(image_paths[j])
        
        ax1.imshow(img1)
        ax1.set_title(f"Image 1: {os.path.basename(image_paths[i])}")
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.set_title(f"Image 2: {os.path.basename(image_paths[j])}")
        ax2.axis('off')
        
        plt.suptitle(f"{method} Pair {idx+1}: Similarity = {sim:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_pair_{idx+1}.png"))
        plt.close()

def count_matching_keypoints(img1_path, img2_path):
    """2つの画像間のマッチするキーポイント数を数える（SIFT特徴量）"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0
    
    # SIFT特徴検出器
    sift = cv2.SIFT_create()
    
    # キーポイントと特徴量を抽出
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0
    
    # BFMatcherでマッチング
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Loweの比率テストでよいマッチだけを選択
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return len(good_matches)

def evaluate_pairs(image_paths, clip_best_pairs, sift_best_pairs, output_dir):
    """選択されたペアの評価を行う"""
    results = {
        'CLIP': [],
        'SIFT': []
    }
    
    # マッチするキーポイント数を評価
    print("Evaluating pairs with keypoint matching...")
    
    for method, pairs in [('CLIP', clip_best_pairs), ('SIFT', sift_best_pairs)]:
        for idx, (i, j, sim) in enumerate(tqdm(pairs, desc=f"{method} Pairs")):
            img1_path = image_paths[i]
            img2_path = image_paths[j]
            
            # マッチするキーポイント数をカウント
            match_count = count_matching_keypoints(img1_path, img2_path)
            
            # 結果を保存
            results[method].append({
                'pair_idx': (i, j),
                'similarity': sim,
                'matching_keypoints': match_count,
                'images': (os.path.basename(img1_path), os.path.basename(img2_path))
            })
    
    # 結果を表にまとめる
    fig, ax = plt.subplots(figsize=(12, 6))
    
    clip_kp = [r['matching_keypoints'] for r in results['CLIP']]
    sift_kp = [r['matching_keypoints'] for r in results['SIFT']]
    clip_sim = [r['similarity'] for r in results['CLIP']]
    sift_sim = [r['similarity'] for r in results['SIFT']]
    
    indices = np.arange(max(len(clip_kp), len(sift_kp)))
    bar_width = 0.35
    
    clip_bars = ax.bar(indices - bar_width/2, clip_kp[:len(indices)], bar_width, label='CLIP', color='blue', alpha=0.7)
    sift_bars = ax.bar(indices + bar_width/2, sift_kp[:len(indices)], bar_width, label='SIFT', color='orange', alpha=0.7)
    
    ax.set_xlabel('Pair Index')
    ax.set_ylabel('Matching Keypoints')
    ax.set_title('Matching Keypoints for Selected Pairs')
    ax.set_xticks(indices)
    ax.legend()
    
    # 各バーに類似度を注釈として追加
    for i, (clip_bar, sift_bar) in enumerate(zip(clip_bars[:len(indices)], sift_bars[:len(indices)])):
        if i < len(clip_sim):
            ax.text(clip_bar.get_x() + clip_bar.get_width()/2, clip_bar.get_height() + 5,
                    f'{clip_sim[i]:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)
        if i < len(sift_sim):
            ax.text(sift_bar.get_x() + sift_bar.get_width()/2, sift_bar.get_height() + 5,
                    f'{sift_sim[i]:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pairs_evaluation.png"))
    plt.close()
    
    # 詳細な結果をテキストファイルに書き出し
    with open(os.path.join(output_dir, "pairs_evaluation.txt"), 'w') as f:
        f.write("=== CLIP Selected Pairs ===\n")
        for idx, r in enumerate(results['CLIP']):
            f.write(f"Pair {idx+1}: {r['images'][0]} - {r['images'][1]}\n")
            f.write(f"  Similarity: {r['similarity']:.4f}\n")
            f.write(f"  Matching Keypoints: {r['matching_keypoints']}\n\n")
            
        f.write("\n=== SIFT Selected Pairs ===\n")
        for idx, r in enumerate(results['SIFT']):
            f.write(f"Pair {idx+1}: {r['images'][0]} - {r['images'][1]}\n")
            f.write(f"  Similarity: {r['similarity']:.4f}\n")
            f.write(f"  Matching Keypoints: {r['matching_keypoints']}\n\n")
    
    return results

def compare_common_pairs(clip_best_pairs, sift_best_pairs, image_paths):
    """CLIPとSIFTで選択された共通のペアを分析"""
    clip_pairs_set = {(min(i, j), max(i, j)) for i, j, _ in clip_best_pairs}
    sift_pairs_set = {(min(i, j), max(i, j)) for i, j, _ in sift_best_pairs}
    
    common_pairs = clip_pairs_set.intersection(sift_pairs_set)
    
    print(f"\nCommon pairs selected by both methods: {len(common_pairs)} out of {len(clip_best_pairs)} CLIP pairs and {len(sift_best_pairs)} SIFT pairs")
    
    if common_pairs:
        print("Common pairs:")
        for i, j in common_pairs:
            clip_sim = next(sim for x, y, sim in clip_best_pairs if (min(x, y), max(x, y)) == (i, j))
            sift_sim = next(sim for x, y, sim in sift_best_pairs if (min(x, y), max(x, y)) == (i, j))
            print(f"  Images: {os.path.basename(image_paths[i])} - {os.path.basename(image_paths[j])}")
            print(f"  CLIP similarity: {clip_sim:.4f}, SIFT similarity: {sift_sim:.4f}")
    
    return common_pairs

def normalize_similarity_map_for_visualization(similarity_matrix):
    """類似度行列を視覚化用に正規化（最大値を1にスケーリング）"""
    if np.max(similarity_matrix) > 0:
        return similarity_matrix / np.max(similarity_matrix)
    return similarity_matrix

def main():
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 画像データのパスを取得
    image_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(image_dir):
        # imagesサブディレクトリがない場合は直接指定されたディレクトリを使用
        image_dir = args.data_dir
    
    # 画像パスの取得
    image_paths = get_image_paths(image_dir, args.sample_size)
    print(f"Testing with {len(image_paths)} images")
    
    # 類似度行列の計算
    clip_selector, sift_selector, clip_similarity, sift_similarity = compute_similarity_matrices(
        image_paths, args.min_overlap, args.max_overlap, args.output_dir
    )
    
    # それぞれの手法で最良ペアを選択
    clip_best_pairs = select_best_pairs(clip_similarity, image_paths, args.min_overlap, args.max_overlap)
    sift_best_pairs = select_best_pairs(sift_similarity, image_paths, args.min_overlap, args.max_overlap)
    
    print(f"\nSelected {len(clip_best_pairs)} pairs using CLIP")
    print(f"Selected {len(sift_best_pairs)} pairs using SIFT")
    
    # 選択されたペアを可視化
    visualize_selected_pairs(image_paths, clip_best_pairs, "CLIP", args.output_dir)
    visualize_selected_pairs(image_paths, sift_best_pairs, "SIFT", args.output_dir)
    
    # 選択されたペアの評価
    evaluation_results = evaluate_pairs(image_paths, clip_best_pairs, sift_best_pairs, args.output_dir)
    
    # 共通のペアを分析
    common_pairs = compare_common_pairs(clip_best_pairs, sift_best_pairs, image_paths)
    
    # 詳細なレポートを生成
    with open(os.path.join(args.output_dir, "summary_report.txt"), 'w') as f:
        f.write("=== Initial Pair Selection Test Report ===\n\n")
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Number of Images: {len(image_paths)}\n")
        f.write(f"Min Overlap: {args.min_overlap}\n")
        f.write(f"Max Overlap: {args.max_overlap}\n\n")
        
        f.write("--- CLIP Method ---\n")
        f.write(f"Feature Type: {clip_selector.feature_type}\n")
        f.write(f"Number of Selected Pairs: {len(clip_best_pairs)}\n")
        if clip_best_pairs:
            f.write(f"Top Pair Similarity: {clip_best_pairs[0][2]:.4f}\n")
            f.write(f"Top Pair Images: {os.path.basename(image_paths[clip_best_pairs[0][0]])} - {os.path.basename(image_paths[clip_best_pairs[0][1]])}\n\n")
        
        f.write("--- SIFT Method ---\n")
        f.write(f"Feature Type: {sift_selector.feature_type}\n")
        f.write(f"Vocabulary Size: {sift_selector.vocab_size}\n")
        f.write(f"Number of Selected Pairs: {len(sift_best_pairs)}\n")
        if sift_best_pairs:
            f.write(f"Top Pair Similarity: {sift_best_pairs[0][2]:.4f}\n")
            f.write(f"Top Pair Images: {os.path.basename(image_paths[sift_best_pairs[0][0]])} - {os.path.basename(image_paths[sift_best_pairs[0][1]])}\n\n")
        
        f.write(f"Common Pairs: {len(common_pairs)} out of {max(len(clip_best_pairs), len(sift_best_pairs))}\n")
        
        f.write("\n--- Detailed Evaluation ---\n")
        for method in ['CLIP', 'SIFT']:
            f.write(f"\n{method} Selected Pairs:\n")
            for idx, r in enumerate(evaluation_results[method]):
                f.write(f"Pair {idx+1}: {r['images'][0]} - {r['images'][1]}\n")
                f.write(f"  Similarity: {r['similarity']:.4f}\n")
                f.write(f"  Matching Keypoints: {r['matching_keypoints']}\n")
    
    print(f"\nTest completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()