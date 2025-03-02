# src/selector/view_selector.py
import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from src.primitive.twod_gaussians_rs import TwoDGaussians

class ViewSelector:
    """
    新規視点選択のためのクラス。
    Bag of Visual Words アプローチと湧出ガウス情報を活用して、
    最適な次の視点を選択します。
    """
    
    def __init__(
        self, 
        image_dir: str,
        vocab_size: int = 200,
        feature_type: str = 'sift',
        min_overlap_ratio: float = 0.2,
        max_overlap_ratio: float = 0.6,
        device: str = 'cpu'
    ):
        """
        ViewSelector の初期化
        
        Args:
            image_dir: 画像が格納されているディレクトリ
            vocab_size: ビジュアルワード（特徴クラスタ）の数
            feature_type: 特徴抽出タイプ ('sift', 'orb' など)
            min_overlap_ratio: 既存視点との最小重なり率
            max_overlap_ratio: 既存視点との最大重なり率（多様性確保のため）
            device: 計算に使用するデバイス
        """
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.feature_type = feature_type
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.device = device
        
        # 初期化
        self.codebook = None  # K-means クラスタリングモデル
        self.image_histograms = {}  # 各画像のBoVWヒストグラム
        self.image_features = {}  # 各画像のオリジナル特徴量とキーポイント
        self.reference_images = []  # 既知の参照視点
        self.source_gaussians_data = None  # 湧出ガウス情報
        
        # 特徴抽出器の初期化
        if feature_type == 'sift':
            self.feature_extractor = cv2.SIFT_create()
        elif feature_type == 'orb':
            self.feature_extractor = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def extract_features(self, image_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """画像から特徴量を抽出"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 特徴量抽出
        keypoints, descriptors = self.feature_extractor.detectAndCompute(img, None)
        
        return keypoints, descriptors
    
    def process_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        複数の画像から特徴量を抽出し、特徴空間を構築する
        
        Args:
            image_paths: 処理する画像のパスリスト
            
        Returns:
            Dict: {画像パス: 特徴量} の辞書
        """
        all_features = []
        features_dict = {}
        
        for path in image_paths:
            try:
                keypoints, descriptors = self.extract_features(path)
                if descriptors is not None and len(descriptors) > 0:
                    features_dict[path] = {
                        'keypoints': keypoints,
                        'descriptors': descriptors
                    }
                    all_features.append(descriptors)
                else:
                    print(f"No features found in {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        # 全ての特徴量を連結
        if all_features:
            combined_features = np.vstack(all_features)
            print(f"Combined {len(combined_features)} features from {len(image_paths)} images")
            self.image_features = features_dict
            return combined_features
        else:
            raise ValueError("No valid features extracted from any image")
    
    def build_codebook(self, features: np.ndarray) -> None:
        """
        特徴量からコードブックを構築
        
        Args:
            features: 全画像から抽出した特徴量の集合
        """
        print(f"Building codebook with {self.vocab_size} clusters...")
        
        # 特徴量が多い場合は MiniBatchKMeans を使用
        if len(features) > 100000:
            self.codebook = MiniBatchKMeans(
                n_clusters=self.vocab_size,
                batch_size=2000,
                random_state=42
            )
        else:
            self.codebook = KMeans(
                n_clusters=self.vocab_size,
                random_state=42,
                n_init=10
            )
        
        # クラスタリング実行
        self.codebook.fit(features)
        print("Codebook built successfully")
    
    def compute_image_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """
        特徴量をコードブックに対するヒストグラムに変換
        
        Args:
            descriptors: 画像から抽出した特徴量
            
        Returns:
            np.ndarray: ビジュアルワードのヒストグラム
        """
        if self.codebook is None:
            raise ValueError("Codebook not built yet. Call build_codebook first.")
        
        # 各特徴量の最も近いクラスタを予測
        predicted_clusters = self.codebook.predict(descriptors)
        
        # ヒストグラム作成
        histogram = np.zeros(self.vocab_size, dtype=np.float32)
        for cluster_id in predicted_clusters:
            histogram[cluster_id] += 1
        
        # ヒストグラム正規化
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        return histogram
    
    def build_histograms(self, image_paths: List[str]) -> None:
        """
        複数の画像のヒストグラムを計算して保存
        
        Args:
            image_paths: ヒストグラムを計算する画像のパスリスト
        """
        for path in image_paths:
            if path in self.image_features:
                descriptors = self.image_features[path]['descriptors']
                histogram = self.compute_image_histogram(descriptors)
                self.image_histograms[path] = histogram
            else:
                print(f"Features for {path} not found. Skipping histogram computation.")
    
    def initialize_from_images(self, image_paths: List[str]) -> None:
        """
        画像セットからコードブックとヒストグラムを初期化
        
        Args:
            image_paths: 初期化に使用する画像のパスリスト
        """
        # 特徴抽出
        combined_features = self.process_images(image_paths)
        
        # コードブック構築
        self.build_codebook(combined_features)
        
        # ヒストグラム計算
        self.build_histograms(image_paths)
        
        print(f"Initialized from {len(image_paths)} images")
    
    def add_reference_images(self, image_names: List[str]) -> None:
        """
        既知の参照視点を追加
        
        Args:
            image_names: 参照視点として追加する画像名のリスト
        """
        for name in image_names:
            # フルパスに変換
            path = os.path.join(self.image_dir, name)
            if path in self.image_histograms:
                self.reference_images.append(path)
            else:
                # ヒストグラムが未計算なら計算
                try:
                    keypoints, descriptors = self.extract_features(path)
                    if descriptors is not None and len(descriptors) > 0:
                        self.image_features[path] = {
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        }
                        histogram = self.compute_image_histogram(descriptors)
                        self.image_histograms[path] = histogram
                        self.reference_images.append(path)
                    else:
                        print(f"No features found in {path}")
                except Exception as e:
                    print(f"Error processing reference image {path}: {e}")
        
        print(f"Added {len(self.reference_images)} reference images")
    
    def set_source_gaussians_data(self, source_gaussians_data: Dict) -> None:
        """
        湧出ガウス情報を設定
        
        Args:
            source_gaussians_data: 湧出ガウス情報を含む辞書
        """
        self.source_gaussians_data = source_gaussians_data
        
        # 湧出ガウスの特徴量表現を計算
        if self.source_gaussians_data is not None:
            self.source_features = {}
            
            # 画像1の湧出ガウス
            if 'source_gaussians1_data' in self.source_gaussians_data:
                data = self.source_gaussians_data['source_gaussians1_data']
                if 'means' in data:
                    # 位置情報を特徴量として使用（単純化）
                    self.source_features['image1'] = data['means']
            
            # 画像2の湧出ガウス
            if 'source_gaussians2_data' in self.source_gaussians_data:
                data = self.source_gaussians_data['source_gaussians2_data']
                if 'means' in data:
                    self.source_features['image2'] = data['means']
    
    def compute_similarity_matrix(self, candidate_paths: List[str]) -> np.ndarray:
        """
        候補画像と参照画像の類似度行列を計算
        
        Args:
            candidate_paths: 候補画像のパスリスト
            
        Returns:
            np.ndarray: 類似度行列 [candidates × references]
        """
        if not self.reference_images:
            raise ValueError("No reference images added. Call add_reference_images first.")
        
        # 候補画像のヒストグラム
        candidate_hists = np.array([self.image_histograms[path] for path in candidate_paths])
        
        # 参照画像のヒストグラム
        reference_hists = np.array([self.image_histograms[path] for path in self.reference_images])
        
        # コサイン類似度計算
        similarity_matrix = cosine_similarity(candidate_hists, reference_hists)
        
        return similarity_matrix
    
    def select_next_view(self, candidate_names: List[str], n_select: int = 1) -> List[str]:
        """
        次の視点を選択
        
        戦略:
        1. 既存視点との一定の重なりを確保（min_overlap_ratio以上）
        2. 湧出ガウスをカバーする可能性が高い画像を優先
        3. 視点の多様性を確保（max_overlap_ratio未満）
        
        Args:
            candidate_names: 候補画像名のリスト
            n_select: 選択する画像の数
            
        Returns:
            List[str]: 選択された画像名のリスト
        """
        # フルパスに変換
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # 未処理の画像を特徴抽出・ヒストグラム計算
        for path in candidate_paths:
            if path not in self.image_histograms:
                try:
                    keypoints, descriptors = self.extract_features(path)
                    if descriptors is not None and len(descriptors) > 0:
                        self.image_features[path] = {
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        }
                        histogram = self.compute_image_histogram(descriptors)
                        self.image_histograms[path] = histogram
                    else:
                        print(f"No features found in {path}, removing from candidates")
                        candidate_paths.remove(path)
                except Exception as e:
                    print(f"Error processing candidate {path}: {e}")
                    candidate_paths.remove(path)
        
        if not candidate_paths:
            raise ValueError("No valid candidate images to select from")
        
        # 類似度行列計算
        similarity_matrix = self.compute_similarity_matrix(candidate_paths)
        
        # 各候補の最大類似度と平均類似度を計算
        max_similarities = np.max(similarity_matrix, axis=1)
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # 選択基準1: 既存視点との重なりが min_overlap_ratio 以上
        overlap_mask = max_similarities >= self.min_overlap_ratio
        # 選択基準2: 重なりすぎない（多様性確保）
        diversity_mask = max_similarities < self.max_overlap_ratio
        # 両方の条件を満たす候補
        valid_candidates = overlap_mask & diversity_mask
        
        # 有効な候補がない場合、最小重なり条件だけで選択
        if not np.any(valid_candidates):
            valid_candidates = overlap_mask
            print("Warning: No candidates satisfy both overlap and diversity criteria. Using only overlap criteria.")
        
        # 有効な候補が依然としてない場合、全候補から選択
        if not np.any(valid_candidates):
            valid_candidates = np.ones_like(overlap_mask, dtype=bool)
            print("Warning: No candidates satisfy overlap criteria. Using all candidates.")
        
        # 有効な候補のインデックス
        valid_indices = np.where(valid_candidates)[0]
        
        # スコア計算（湧出ガウス情報をもとに）
        scores = np.zeros(len(valid_indices))
        
        # 基本スコア: 既存視点との類似度（適度な重なりを持つものが良い）
        normalized_similarities = (max_similarities[valid_indices] - self.min_overlap_ratio) / (self.max_overlap_ratio - self.min_overlap_ratio)
        normalized_similarities = np.clip(normalized_similarities, 0, 1)
        
        # 中程度の類似度（0.5付近）が最もスコアが高くなるよう設定
        similarity_scores = 1.0 - 2.0 * np.abs(normalized_similarities - 0.5)
        scores += similarity_scores
        
        # 湧出ガウス情報を利用したスコア付け
        if self.source_gaussians_data is not None and hasattr(self, 'source_features'):
            for img_idx, candidate_idx in enumerate(valid_indices):
                candidate_path = candidate_paths[candidate_idx]
                
                # 候補画像の特徴点位置
                if candidate_path in self.image_features:
                    candidate_keypoints = self.image_features[candidate_path]['keypoints']
                    keypoint_positions = np.array([kp.pt for kp in candidate_keypoints])
                    
                    # 湧出ガウスとの位置的類似度
                    source_score = 0.0
                    for src_key, src_positions in self.source_features.items():
                        # 単純な実装: 特徴点の位置分布の類似度
                        if len(keypoint_positions) > 0 and len(src_positions) > 0:
                            # 画像サイズを取得
                            img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            height, width = img.shape[:2]
                            
                            # スケール正規化（画像サイズを考慮）
                            norm_kp = keypoint_positions / np.array([width, height])
                            
                            # PyTorchテンソルをNumPy配列に変換
                            if hasattr(src_positions, 'numpy'):
                                src_positions_np = src_positions.numpy()
                            else:
                                src_positions_np = np.array(src_positions)
                                
                            norm_src = src_positions_np / np.array([width, height])
                            
                            # 単純な分布類似度: 平均と分散の差
                            kp_mean = np.mean(norm_kp, axis=0)
                            src_mean = np.mean(norm_src, axis=0)
                            mean_diff = np.linalg.norm(kp_mean - src_mean)
                            
                            kp_var = np.var(norm_kp, axis=0)
                            src_var = np.var(norm_src, axis=0)
                            var_diff = np.linalg.norm(kp_var - src_var)
                            
                            # スコア計算（差分が小さいほど高スコア）
                            src_score = 1.0 / (1.0 + 10.0 * (mean_diff + var_diff))
                            source_score += src_score
                    
                    # 湧出ガウススコアを加算（ソースごとの平均）
                    if len(self.source_features) > 0:
                        source_score /= len(self.source_features)
                        scores[img_idx] += 2.0 * source_score  # 湧出ガウス対応を優先
        
        # 最高スコアの候補を選択
        best_indices = np.argsort(-scores)[:n_select]
        selected_indices = [valid_indices[i] for i in best_indices]
        selected_paths = [candidate_paths[i] for i in selected_indices]
        
        # パスから画像名に変換して返却
        selected_names = [os.path.basename(path) for path in selected_paths]
        
        # 選択結果のデバッグ情報
        print(f"Selected {len(selected_names)} images:")
        for i, name in enumerate(selected_names):
            idx = selected_indices[i]
            print(f"  {name}: similarity={max_similarities[idx]:.3f}, avg_sim={avg_similarities[idx]:.3f}, score={scores[best_indices[i]]:.3f}")
        
        return selected_names
    
    def visualize_selection(self, selected_names: List[str], candidate_names: List[str]) -> None:
        """選択結果を可視化"""
        # フルパスに変換
        selected_paths = [os.path.join(self.image_dir, name) for name in selected_names]
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # 類似度行列
        similarity_matrix = self.compute_similarity_matrix(candidate_paths)
        
        # 選択された画像のインデックス
        selected_indices = [candidate_paths.index(path) for path in selected_paths if path in candidate_paths]
        
        # 可視化
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 候補全体の類似度分布
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # ヒストグラム
        ax.hist(max_similarities, bins=20, alpha=0.5, label='All Candidates')
        
        # 選択された画像の類似度
        if selected_indices:
            selected_similarities = max_similarities[selected_indices]
            ax.hist(selected_similarities, bins=10, alpha=0.7, label='Selected Views')
            
            # 選択された各画像の位置を表示
            for i, idx in enumerate(selected_indices):
                sim = max_similarities[idx]
                ax.axvline(x=sim, color='r', linestyle='--', alpha=0.7)
                ax.text(sim, 0, f" {selected_names[i]}", rotation=90, verticalalignment='bottom')
        
        # 閾値ライン
        ax.axvline(x=self.min_overlap_ratio, color='g', linestyle='-', label=f'Min Overlap ({self.min_overlap_ratio})')
        ax.axvline(x=self.max_overlap_ratio, color='r', linestyle='-', label=f'Max Overlap ({self.max_overlap_ratio})')
        
        ax.set_xlabel('Maximum Similarity to Reference Views')
        ax.set_ylabel('Number of Candidates')
        ax.set_title('View Selection: Similarity Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/view_selection_distribution.png')
        plt.close()
        
        print(f"Selection visualization saved to results/view_selection_distribution.png")
        
        

    def compute_gaussian_similarity(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians) -> float:
        """
        2D Gaussianのdistrubution特性に基づく類似度を計算
        
        Args:
            gaussians1: 1つ目の2D Gaussians
            gaussians2: 2つ目の2D Gaussians
            
        Returns:
            float: 類似度スコア (0〜1)
        """
        # 1. 中心点の空間分布の類似度
        means1 = gaussians1.means
        means2 = gaussians2.means
        
        if hasattr(means1, 'detach'):
            means1 = means1.detach().cpu().numpy()
        if hasattr(means2, 'detach'):
            means2 = means2.detach().cpu().numpy()
            
        # 分布の中心と分散を比較
        center1 = np.mean(means1, axis=0)
        center2 = np.mean(means2, axis=0)
        center_dist = np.linalg.norm(center1 - center2) / np.linalg.norm(center1 + center2 + 1e-6)
        
        var1 = np.var(means1, axis=0)
        var2 = np.var(means2, axis=0)
        var_ratio = np.mean(np.maximum(var1, var2) / np.maximum(np.minimum(var1, var2), 1e-6))
        var_score = 1.0 / (1.0 + np.log(1 + var_ratio))
        
        # 2. スケール分布の類似度
        scales1 = gaussians1.scales
        scales2 = gaussians2.scales
        
        if hasattr(scales1, 'detach'):
            scales1 = scales1.detach().cpu().numpy()
        if hasattr(scales2, 'detach'):
            scales2 = scales2.detach().cpu().numpy()
        
        scale1 = np.mean(scales1, axis=0)
        scale2 = np.mean(scales2, axis=0)
        scale_ratio = np.mean(np.maximum(scale1, scale2) / np.maximum(np.minimum(scale1, scale2), 1e-6))
        scale_score = 1.0 / (1.0 + np.log(1 + scale_ratio))
        
        # 3. 色分布の類似度
        rgb1 = gaussians1.rgb
        rgb2 = gaussians2.rgb
        
        if hasattr(rgb1, 'detach'):
            rgb1 = rgb1.detach().cpu().numpy()
        if hasattr(rgb2, 'detach'):
            rgb2 = rgb2.detach().cpu().numpy()
        
        # 色ヒストグラムの類似度（簡易的な実装）
        hist1, _ = np.histogramdd(rgb1, bins=8, range=[[0, 1], [0, 1], [0, 1]])
        hist2, _ = np.histogramdd(rgb2, bins=8, range=[[0, 1], [0, 1], [0, 1]])
        
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        color_sim = np.sum(np.minimum(hist1, hist2))
        
        # 4. ガウスの空間的な分布の広がり（多様性）
        diversity1 = np.sqrt(np.sum(var1))
        diversity2 = np.sqrt(np.sum(var2))
        diversity_ratio = min(diversity1, diversity2) / max(diversity1, diversity2)
        
        # 最終的な類似度スコアの計算
        final_score = (
            0.2 * (1.0 - center_dist) +  # 中心位置の類似度
            0.3 * var_score +            # 分散の類似度
            0.2 * scale_score +          # スケールの類似度
            0.2 * color_sim +            # 色分布の類似度
            0.1 * diversity_ratio        # 空間的な多様性の類似度
        )
        
        return final_score

    def estimate_view_angle_change(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians) -> float:
        """
        2つのガウス集合から画角変化を推定し、スコア化
        
        Args:
            gaussians1: 1つ目の2D Gaussians
            gaussians2: 2つ目の2D Gaussians
            
        Returns:
            float: 画角変化のスコア (0〜1, 高いほど変化が少ない)
        """
        # ガウス分布の広がり（共分散行列）の比較
        covs1 = gaussians1.covs
        covs2 = gaussians2.covs
        
        if hasattr(covs1, 'detach'):
            covs1 = covs1.detach().cpu().numpy()
        if hasattr(covs2, 'detach'):
            covs2 = covs2.detach().cpu().numpy()
        
        # 共分散行列の行列式（面積）の平均を計算
        det1 = np.mean([np.linalg.det(cov) for cov in covs1])
        det2 = np.mean([np.linalg.det(cov) for cov in covs2])
        
        # 行列式の比率（大きいほど画角変化が大きい）
        ratio = max(det1, det2) / max(min(det1, det2), 1e-10)
        
        # 画角変化のスコア（0〜1, 高いほど変化が少ない）
        return 1.0 / (1.0 + np.log(1 + ratio))

    def compute_initial_pair_score_gs(self, 
                                img1_name: str, 
                                img2_name: str, 
                                gaussians1: TwoDGaussians,
                                gaussians2: TwoDGaussians,
                                feature_sim: float) -> float:
        """
        初期ペア選択のためのスコアを計算
        
        Args:
            img1_name: 1つ目の画像名
            img2_name: 2つ目の画像名
            gaussians1: 1つ目の2D Gaussians
            gaussians2: 2つ目の2D Gaussians
            feature_sim: BoVWによる特徴量類似度
            
        Returns:
            float: 最終スコア
        """
        # 1. ガウスベースの類似度
        gauss_sim = self.compute_gaussian_similarity(gaussians1, gaussians2)
        
        # 2. 画角変化のスコア
        angle_score = self.estimate_view_angle_change(gaussians1, gaussians2)
        
        # 3. ガウス数（ガウスの豊富さ）のスコア
        count_ratio = min(gaussians1.k, gaussians2.k) / max(gaussians1.k, gaussians2.k)
        
        # 最終スコアの計算
        final_score = (
            0.3 * feature_sim +   # BoVW特徴量類似度
            0.4 * gauss_sim +     # ガウス分布の類似度
            0.2 * angle_score +   # 画角変化の少なさ
            0.1 * count_ratio     # ガウス数の均衡
        )
        
        print(f"Pair {img1_name}-{img2_name}: feature_sim={feature_sim:.3f}, gauss_sim={gauss_sim:.3f}, "
            f"angle_score={angle_score:.3f}, count_ratio={count_ratio:.3f}, final={final_score:.3f}")
        
        return final_score