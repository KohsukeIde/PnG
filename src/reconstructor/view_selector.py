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
    Class for selecting new viewpoints.
    Uses Bag of Visual Words approach and source Gaussian information
    to select the optimal next viewpoint.
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
        Initialize ViewSelector
        
        Args:
            image_dir: Directory containing images
            vocab_size: Number of visual words (feature clusters)
            feature_type: Feature extraction type ('sift', 'orb', etc.)
            min_overlap_ratio: Minimum overlap ratio with existing views
            max_overlap_ratio: Maximum overlap ratio with existing views (for diversity)
            device: Device to use for computation
        """
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.feature_type = feature_type
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.device = device
        
        # Initialization
        self.codebook = None  # K-means clustering model
        self.image_histograms = {}  # BoVW histograms for each image
        self.image_features = {}  # Original features and keypoints for each image
        self.reference_images = []  # Known reference viewpoints
        self.source_gaussians_data = None  # Source Gaussian information
        
        # Initialize feature extractor
        if feature_type == 'sift':
            self.feature_extractor = cv2.SIFT_create()
        elif feature_type == 'orb':
            self.feature_extractor = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def extract_features(self, image_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract features from an image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Extract features
        keypoints, descriptors = self.feature_extractor.detectAndCompute(img, None)
        
        return keypoints, descriptors
    
    def process_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """Extract features from multiple images and build feature space
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            Dict: Dictionary of {image_path: features}
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
        
        # Concatenate all features
        if all_features:
            combined_features = np.vstack(all_features)
            print(f"Combined {len(combined_features)} features from {len(image_paths)} images")
            self.image_features = features_dict
            return combined_features
        else:
            raise ValueError("No valid features extracted from any image")
    
    def build_codebook(self, features: np.ndarray) -> None:
        """
        Build codebook from features
        
        Args:
            features: Collection of features extracted from all images
        """
        print(f"Building codebook with {self.vocab_size} clusters...")
        
        # Use MiniBatchKMeans for large feature sets
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
        
        # Perform clustering
        self.codebook.fit(features)
        print("Codebook built successfully")
    
    def compute_image_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert features to histogram against codebook
        
        Args:
            descriptors: Features extracted from an image
            
        Returns:
            np.ndarray: Histogram of visual words
        """
        if self.codebook is None:
            raise ValueError("Codebook not built yet. Call build_codebook first.")
        
        # Predict closest cluster for each feature
        predicted_clusters = self.codebook.predict(descriptors)
        
        # Create histogram
        histogram = np.zeros(self.vocab_size, dtype=np.float32)
        for cluster_id in predicted_clusters:
            histogram[cluster_id] += 1
        
        # Normalize histogram　(特徴点の総数に依存させないため)
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        return histogram
    
    def build_histograms(self, image_paths: List[str]) -> None:
        """
        Compute and store histograms for multiple images
        
        Args:
            image_paths: List of image paths to compute histograms for
        """
        for path in image_paths:
            if path in self.image_features:
                descriptors = self.image_features[path]['descriptors']
                histogram = self.compute_image_histogram(descriptors)
                self.image_histograms[path] = histogram
            else:
                print(f"Features for {path} not found. Skipping histogram computation.")
    
    def initialize_from_images(self, image_paths: List[str]) -> None:
        """Initialize codebook and histograms from a set of images
        
        Args:
            image_paths: List of image paths to use for initialization
        """
        # Extract features
        combined_features = self.process_images(image_paths)
        
        # Build codebook
        self.build_codebook(combined_features)
        
        # Compute histograms
        self.build_histograms(image_paths)
        
        print(f"Initialized from {len(image_paths)} images")
    
    def add_reference_images(self, image_names: List[str]) -> None:
        """ Add known reference viewpoints
        
        Args:
            image_names: List of image names to add as reference viewpoints
        """
        for name in image_names:
            # Convert to full path
            path = os.path.join(self.image_dir, name)
            if path in self.image_histograms:
                self.reference_images.append(path)
            else:
                # Compute histogram if not already calculated
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
        Set source Gaussian information
        
        Args:
            source_gaussians_data: Dictionary containing source Gaussian information
        """
        self.source_gaussians_data = source_gaussians_data
        
        # Calculate feature representation of source Gaussians
        if self.source_gaussians_data is not None:
            self.source_features = {}
            
            # Source Gaussians from image 1
            if 'source_gaussians1_data' in self.source_gaussians_data:
                data = self.source_gaussians_data['source_gaussians1_data']
                if 'means' in data:
                    # Use position information as features (simplified)
                    self.source_features['image1'] = data['means']
            
            # Source Gaussians from image 2
            if 'source_gaussians2_data' in self.source_gaussians_data:
                data = self.source_gaussians_data['source_gaussians2_data']
                if 'means' in data:
                    self.source_features['image2'] = data['means']
    
    def compute_similarity_matrix(self, candidate_paths: List[str]) -> np.ndarray:
        """Compute similarity matrix between candidate images and reference images
        
        Args:
            candidate_paths: List of candidate image paths
            
        Returns:
            np.ndarray: Similarity matrix [candidates × references]
        """
        if not self.reference_images:
            raise ValueError("No reference images added. Call add_reference_images first.")
        
        # Histograms for candidate images
        candidate_hists = np.array([self.image_histograms[path] for path in candidate_paths])
        
        # Histograms for reference images
        reference_hists = np.array([self.image_histograms[path] for path in self.reference_images])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(candidate_hists, reference_hists)
        
        return similarity_matrix
    
    def select_next_view(self, candidate_names: List[str], n_select: int = 1) -> List[str]:
        """Select next viewpoint
        
        Strategy:
        1. Ensure sufficient overlap with existing views (at least min_overlap_ratio)
        2. Prioritize images likely to cover source Gaussians
        3. Ensure viewpoint diversity (less than max_overlap_ratio)
        
        Args:
            candidate_names: List of candidate image names
            n_select: Number of images to select
            
        Returns:
            List[str]: List of selected image names
        """
        # Convert to full paths
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # Process unprocessed images
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
        
        # Calculate similarity matrix
        similarity_matrix = self.compute_similarity_matrix(candidate_paths)
        
        # Calculate maximum and average similarities for each candidate
        max_similarities = np.max(similarity_matrix, axis=1)
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Selection criterion 1: Overlap with existing views >= min_overlap_ratio
        overlap_mask = max_similarities >= self.min_overlap_ratio
        # Selection criterion 2: Not too much overlap (ensure diversity)
        diversity_mask = max_similarities < self.max_overlap_ratio
        # Candidates satisfying both conditions
        valid_candidates = overlap_mask & diversity_mask
        
        # If no valid candidates, use only overlap criterion
        if not np.any(valid_candidates):
            valid_candidates = overlap_mask
            print("Warning: No candidates satisfy both overlap and diversity criteria. Using only overlap criteria.")
        
        # If still no valid candidates, use all candidates
        if not np.any(valid_candidates):
            valid_candidates = np.ones_like(overlap_mask, dtype=bool)
            print("Warning: No candidates satisfy overlap criteria. Using all candidates.")
        
        # Indices of valid candidates
        valid_indices = np.where(valid_candidates)[0]
        
        # Calculate scores (based on source Gaussian information)
        scores = np.zeros(len(valid_indices))
        
        # Base score: Similarity to existing views (moderate overlap is preferred)
        normalized_similarities = (max_similarities[valid_indices] - self.min_overlap_ratio) / (self.max_overlap_ratio - self.min_overlap_ratio)
        normalized_similarities = np.clip(normalized_similarities, 0, 1)
        
        # Highest score for moderate similarity (around 0.5)
        similarity_scores = 1.0 - 2.0 * np.abs(normalized_similarities - 0.5)
        scores += similarity_scores
        
        # Score based on source Gaussian information
        if self.source_gaussians_data is not None and hasattr(self, 'source_features'):
            for img_idx, candidate_idx in enumerate(valid_indices):
                candidate_path = candidate_paths[candidate_idx]
                
                # Keypoint positions in candidate image
                if candidate_path in self.image_features:
                    candidate_keypoints = self.image_features[candidate_path]['keypoints']
                    keypoint_positions = np.array([kp.pt for kp in candidate_keypoints])
                    
                    # Positional similarity to source Gaussians
                    source_score = 0.0
                    for src_key, src_positions in self.source_features.items():
                        # Simple implementation: similarity of feature point position distributions
                        if len(keypoint_positions) > 0 and len(src_positions) > 0:
                            # Get image size
                            img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            height, width = img.shape[:2]
                            
                            # Scale normalization (considering image size)
                            norm_kp = keypoint_positions / np.array([width, height])
                            
                            # Convert PyTorch tensor to NumPy array if needed
                            if hasattr(src_positions, 'numpy'):
                                src_positions_np = src_positions.numpy()
                            else:
                                src_positions_np = np.array(src_positions)
                                
                            norm_src = src_positions_np / np.array([width, height])
                            
                            # Simple distribution similarity: difference in mean and variance
                            kp_mean = np.mean(norm_kp, axis=0)
                            src_mean = np.mean(norm_src, axis=0)
                            mean_diff = np.linalg.norm(kp_mean - src_mean)
                            
                            kp_var = np.var(norm_kp, axis=0)
                            src_var = np.var(norm_src, axis=0)
                            var_diff = np.linalg.norm(kp_var - src_var)
                            
                            # Score calculation (smaller difference = higher score)
                            src_score = 1.0 / (1.0 + 10.0 * (mean_diff + var_diff))
                            source_score += src_score
                    
                    # Add source Gaussian score (average across sources)
                    if len(self.source_features) > 0:
                        source_score /= len(self.source_features)
                        scores[img_idx] += 2.0 * source_score  # Prioritize source Gaussian correspondence
        
        # Select candidates with highest scores
        best_indices = np.argsort(-scores)[:n_select]
        selected_indices = [valid_indices[i] for i in best_indices]
        selected_paths = [candidate_paths[i] for i in selected_indices]
        
        # Convert paths to image names for return
        selected_names = [os.path.basename(path) for path in selected_paths]
        
        # Debug information about selection
        print(f"Selected {len(selected_names)} images:")
        for i, name in enumerate(selected_names):
            idx = selected_indices[i]
            print(f"  {name}: similarity={max_similarities[idx]:.3f}, avg_sim={avg_similarities[idx]:.3f}, score={scores[best_indices[i]]:.3f}")
        
        return selected_names
    
    def select_next_view_simple(self, candidate_names: List[str], n_select: int = 1) -> List[str]:
        """既存視点との類似度が最も高い画像を選択する
        
        Args:
            candidate_names: 候補画像名のリスト
            n_select: 選択する画像の数
            
        Returns:
            List[str]: 選択された画像名のリスト
        """
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
        
        # 各候補の最大類似度を計算
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # 類似度が最大のものから順に選択
        best_indices = np.argsort(-max_similarities)[:n_select]
        selected_paths = [candidate_paths[i] for i in best_indices]
        
        selected_names = [os.path.basename(path) for path in selected_paths]
        
        print(f"Selected {len(selected_names)} images based on highest similarity:")
        for i, name in enumerate(selected_names):
            idx = best_indices[i]
            print(f"  {name}: similarity={max_similarities[idx]:.3f}")
        
        return selected_names
    
    def visualize_selection(self, selected_names: List[str], candidate_names: List[str]) -> None:
        """Visualize selection results"""
        # Convert to full paths
        selected_paths = [os.path.join(self.image_dir, name) for name in selected_names]
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # Similarity matrix
        similarity_matrix = self.compute_similarity_matrix(candidate_paths)
        
        # Indices of selected images
        selected_indices = [candidate_paths.index(path) for path in selected_paths if path in candidate_paths]
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Similarity distribution for all candidates
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Histogram
        ax.hist(max_similarities, bins=20, alpha=0.5, label='All Candidates')
        
        # Similarities of selected images
        if selected_indices:
            selected_similarities = max_similarities[selected_indices]
            ax.hist(selected_similarities, bins=10, alpha=0.7, label='Selected Views')
            
            # Show position of each selected image
            for i, idx in enumerate(selected_indices):
                sim = max_similarities[idx]
                ax.axvline(x=sim, color='r', linestyle='--', alpha=0.7)
                ax.text(sim, 0, f" {selected_names[i]}", rotation=90, verticalalignment='bottom')
        
        # Threshold lines
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
        Compute similarity based on 2D Gaussian distribution characteristics
        
        Args:
            gaussians1: First set of 2D Gaussians
            gaussians2: Second set of 2D Gaussians
            
        Returns:
            float: Similarity score (0-1)
        """
        # 1. Similarity of center point spatial distributions
        means1 = gaussians1.means
        means2 = gaussians2.means
        
        if hasattr(means1, 'detach'):
            means1 = means1.detach().cpu().numpy()
        if hasattr(means2, 'detach'):
            means2 = means2.detach().cpu().numpy()
            
        # Compare distribution centers and variances
        center1 = np.mean(means1, axis=0)
        center2 = np.mean(means2, axis=0)
        center_dist = np.linalg.norm(center1 - center2) / np.linalg.norm(center1 + center2 + 1e-6)
        
        var1 = np.var(means1, axis=0)
        var2 = np.var(means2, axis=0)
        var_ratio = np.mean(np.maximum(var1, var2) / np.maximum(np.minimum(var1, var2), 1e-6))
        var_score = 1.0 / (1.0 + np.log(1 + var_ratio))
        
        # 2. Similarity of scale distributions
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
        
        # 3. Similarity of color distributions
        rgb1 = gaussians1.rgb
        rgb2 = gaussians2.rgb
        
        if hasattr(rgb1, 'detach'):
            rgb1 = rgb1.detach().cpu().numpy()
        if hasattr(rgb2, 'detach'):
            rgb2 = rgb2.detach().cpu().numpy()
        
        # Color histogram similarity (simplified implementation)
        hist1, _ = np.histogramdd(rgb1, bins=8, range=[[0, 1], [0, 1], [0, 1]])
        hist2, _ = np.histogramdd(rgb2, bins=8, range=[[0, 1], [0, 1], [0, 1]])
        
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        color_sim = np.sum(np.minimum(hist1, hist2))
        
        # 4. Spatial diversity of Gaussian distributions
        diversity1 = np.sqrt(np.sum(var1))
        diversity2 = np.sqrt(np.sum(var2))
        diversity_ratio = min(diversity1, diversity2) / max(diversity1, diversity2)
        
        # Calculate final similarity score
        final_score = (
            0.2 * (1.0 - center_dist) +  # Center position similarity
            0.3 * var_score +            # Variance similarity
            0.2 * scale_score +          # Scale similarity
            0.2 * color_sim +            # Color distribution similarity
            0.1 * diversity_ratio        # Spatial diversity similarity
        )
        
        return final_score

    def estimate_view_angle_change(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians) -> float:
        """
        Estimate view angle change from two Gaussian sets and convert to score
        
        Args:
            gaussians1: First set of 2D Gaussians
            gaussians2: Second set of 2D Gaussians
            
        Returns:
            float: View angle change score (0-1, higher means less change)
        """
        # Compare Gaussian distribution spread (covariance matrices)
        covs1 = gaussians1.covs
        covs2 = gaussians2.covs
        
        if hasattr(covs1, 'detach'):
            covs1 = covs1.detach().cpu().numpy()
        if hasattr(covs2, 'detach'):
            covs2 = covs2.detach().cpu().numpy()
        
        # Calculate average determinant (area) of covariance matrices
        det1 = np.mean([np.linalg.det(cov) for cov in covs1])
        det2 = np.mean([np.linalg.det(cov) for cov in covs2])
        
        # Determinant ratio (higher means larger view angle change)
        ratio = max(det1, det2) / max(min(det1, det2), 1e-10)
        
        # View angle change score (0-1, higher means less change)
        return 1.0 / (1.0 + np.log(1 + ratio))

    def compute_initial_pair_score_gs(self, 
                                img1_name: str, 
                                img2_name: str, 
                                gaussians1: TwoDGaussians,
                                gaussians2: TwoDGaussians,
                                feature_sim: float) -> float:
        """
        Compute score for initial pair selection
        
        Args:
            img1_name: First image name
            img2_name: Second image name
            gaussians1: First set of 2D Gaussians
            gaussians2: Second set of 2D Gaussians
            feature_sim: Feature similarity from BoVW
            
        Returns:
            float: Final score
        """
        # 1. Gaussian-based similarity
        gauss_sim = self.compute_gaussian_similarity(gaussians1, gaussians2)
        
        # 2. View angle change score
        angle_score = self.estimate_view_angle_change(gaussians1, gaussians2)
        
        # 3. Gaussian count (richness) score
        count_ratio = min(gaussians1.k, gaussians2.k) / max(gaussians1.k, gaussians2.k)
        
        # Calculate final score
        final_score = (
            0.3 * feature_sim +   # BoVW feature similarity
            0.4 * gauss_sim +     # Gaussian distribution similarity
            0.2 * angle_score +   # Low view angle change
            0.1 * count_ratio     # Balanced Gaussian count
        )
        
        print(f"Pair {img1_name}-{img2_name}: feature_sim={feature_sim:.3f}, gauss_sim={gauss_sim:.3f}, "
            f"angle_score={angle_score:.3f}, count_ratio={count_ratio:.3f}, final={final_score:.3f}")
        
        return final_score