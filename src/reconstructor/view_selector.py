# src/selector/view_selector.py
import os
import sys
import numpy as np
import torch
import cv2
import torch
from PIL import Image
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from src.primitive.twod_gaussians_rs import TwoDGaussians

class ViewSelector:
    """
    Class for selecting new viewpoints based on CLIP features and 2D Gaussians.
    For textureless images where traditional feature extractors fail.
    """
    
    def __init__(
        self, 
        image_dir: str,
        vocab_size: int = 200,
        feature_type: str = 'sift',
        min_overlap_ratio: float = 0.2,
        max_overlap_ratio: float = 0.6,
        device: str = 'cpu',
        clip_model_name: str = "ViT-B/32"
    ):
        """
        Initialize ViewSelector with either CLIP features or traditional features
        
        Args:
            image_dir: Directory containing images
            vocab_size: Number of visual words (feature clusters) for SIFT/ORB
            feature_type: Feature extraction type ('clip', 'sift', 'orb')
            min_overlap_ratio: Minimum overlap ratio with existing views
            max_overlap_ratio: Maximum overlap ratio with existing views (for diversity)
            device: Device to use for computation
            clip_model_name: CLIP model name to use for feature extraction
        """
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.feature_type = feature_type.lower()
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.device = device
        self.clip_model_name = clip_model_name
        
        # Initialization
        self.clip_features = {}  # CLIP embeddings for each image
        self.image_features = {}  # Features for traditional methods
        self.image_histograms = {}  # For BoVW approach
        self.codebook = None  # For BoVW clustering
        self.reference_images = []  # Known reference viewpoints
        self.source_gaussians_data = None  # Source Gaussian information
        
        # Flag to track whether CLIP initialization succeeded
        self.clip_model = None
        self.clip_preprocess = None
        clip_initialized = False
        
        # Initialize feature extractor based on type
        if self.feature_type == 'clip':
            try:
                import clip
                
                # Set torch device
                self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Load CLIP model
                print(f"Loading CLIP model {clip_model_name}...")
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.torch_device)
                print(f"CLIP model loaded on {self.torch_device}")
                clip_initialized = True
            except Exception as e:
                print(f"Error loading CLIP: {str(e)}. Falling back to SIFT")
                self.feature_type = 'sift'
        
        # Initialize traditional feature extractors if CLIP failed or not requested
        if not clip_initialized or self.feature_type != 'clip':
            if self.feature_type == 'orb':
                self.feature_extractor = cv2.ORB_create()
                print("Using ORB feature extractor")
            else:
                # Default to SIFT for any other case
                self.feature_type = 'sift'
                self.feature_extractor = cv2.SIFT_create()
                print("Using SIFT feature extractor")
        
    def extract_clip_features(self, image_path: str) -> np.ndarray:
        """Extract CLIP image embeddings from an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: CLIP image embedding or None if CLIP is not available
        """
        # First check if CLIP is enabled and initialized
        if self.feature_type != 'clip' or self.clip_model is None:
            return None
            
        try:
            # Load image using PIL (CLIP requires RGB)
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess image and extract CLIP features
            with torch.no_grad():
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.torch_device)
                image_features = self.clip_model.encode_image(image_input)
                
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting CLIP features from {image_path}: {e}")
            return None
    
    def extract_features(self, image_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract traditional features (SIFT/ORB) from an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple: (keypoints, descriptors)
        """ 
        # Read image in grayscale for feature extraction
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None, None
        
        # Extract features using selected extractor
        keypoints, descriptors = self.feature_extractor.detectAndCompute(img, None)
        
        return keypoints, descriptors

    
    def process_images(self, image_paths: List[str]) -> None:
        """Extract features from multiple images based on feature type
        
        Args:
            image_paths: List of image paths to process
        """
        if self.feature_type == 'clip' and self.clip_model is not None:
            # Process using CLIP features
            for path in image_paths:
                clip_features = self.extract_clip_features(path)
                if clip_features is not None:
                    self.clip_features[path] = clip_features
                else:
                    print(f"Failed to extract CLIP features from {path}")
            
            print(f"Extracted CLIP features from {len(self.clip_features)} images")
            
        else:
            # Process using traditional features (SIFT/ORB)
            all_features = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    keypoints, descriptors = self.extract_features(path)
                    if descriptors is not None and len(descriptors) > 0:
                        self.image_features[path] = {
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        }
                        all_features.append(descriptors)
                        valid_paths.append(path)
                    else:
                        print(f"No features found in {path}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            print(f"Extracted features from {len(self.image_features)} images")
            
            # Build codebook for BoVW only if we have features
            if all_features and len(all_features) > 0:
                try:
                    self.build_codebook(np.vstack(all_features))
                    # Compute histograms
                    self.build_histograms(valid_paths)
                except Exception as e:
                    print(f"Error building codebook: {e}")
    
    def build_codebook(self, features: np.ndarray) -> None:
        """Build codebook (visual vocabulary) from features for BoVW
        
        Args:
            features: Collection of features extracted from all images
        """
        if self.feature_type == 'clip':
            return
            
        print(f"Building codebook with {self.vocab_size} clusters...")
        
        # Use MiniBatchKMeans for large feature sets
        if len(features) > 100000:
            from sklearn.cluster import MiniBatchKMeans
            self.codebook = MiniBatchKMeans(
                n_clusters=self.vocab_size,
                batch_size=2000,
                random_state=42
            )
        else:
            from sklearn.cluster import KMeans
            self.codebook = KMeans(
                n_clusters=self.vocab_size,
                random_state=42,
                n_init=10
            )
        
        # Perform clustering
        self.codebook.fit(features)
        print("Codebook built successfully")
    
    def compute_image_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert features to histogram against codebook for BoVW
        
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
        
        # Normalize histogram
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        return histogram
    
    def build_histograms(self, image_paths: List[str]) -> None:
        """Compute and store histograms for multiple images for BoVW
        
        Args:
            image_paths: List of image paths to compute histograms for
        """
        if self.feature_type == 'clip':
            return
            
        for path in image_paths:
            if path in self.image_features:
                descriptors = self.image_features[path]['descriptors']
                histogram = self.compute_image_histogram(descriptors)
                self.image_histograms[path] = histogram
            else:
                print(f"Features for {path} not found. Skipping histogram computation.")
    
    def add_reference_images(self, image_names: List[str]) -> None:
    
    
    
        """ Add known reference viewpoints
        
        Args:
            image_names: List of image names to add as reference viewpoints
        """
        for name in image_names:
            # Convert to full path
            path = os.path.join(self.image_dir, name)
            
            # Simply add to reference_images if features exist
            if self.feature_type == 'clip':
                if path in self.clip_features:
                    self.reference_images.append(path)
                else:
                    sys.exit(f"Warning: Cannot add {name} as reference - CLIP features not found")
            else:
                # For traditional features
                if path in self.image_histograms:
                    self.reference_images.append(path)
                else:
                    # Try to compute histogram if features exist but histogram doesn't
                    if path in self.image_features and self.codebook is not None:
                        descriptors = self.image_features[path]['descriptors']
                        histogram = self.compute_image_histogram(descriptors)
                        self.image_histograms[path] = histogram
                        self.reference_images.append(path)
                    else:
                        sys.exit(f"Warning: Cannot add {name} as reference - features or histograms not found")
                        
                        
        print(f"Added {len(self.reference_images)} reference images")
        
    def set_source_gaussians_data(self, source_gaussians_data: Dict) -> None:
        """Set source Gaussian information
        
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
                else:
                    sys.exit(f"Warning: Cannot add source gaussians from image 1 - 'means' not found")
            # Source Gaussians from image 2
            if 'source_gaussians2_data' in self.source_gaussians_data:
                data = self.source_gaussians_data['source_gaussians2_data']
                if 'means' in data:
                    self.source_features['image2'] = data['means']
    def compute_similarity_matrix(self, candidate_paths: List[str]) -> np.ndarray:
        """Compute similarity matrix between candidate images and reference images
        based on selected feature type (CLIP or BoVW)
        
        Args:
            candidate_paths: List of candidate image paths
            
        Returns:
            np.ndarray: Similarity matrix [candidates × references]
        """
        if not self.reference_images:
            raise ValueError("No reference images added. Call add_reference_images first.")
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((len(candidate_paths), len(self.reference_images)))
        
        if self.feature_type == 'clip':
            # CLIP similarity using embeddings
            clip_candidate_paths = [p for p in candidate_paths if p in self.clip_features]
            clip_reference_paths = [r for r in self.reference_images if r in self.clip_features]
            
            if clip_candidate_paths and clip_reference_paths:
                # Extract CLIP features
                clip_candidate_features = np.vstack([self.clip_features[p] for p in clip_candidate_paths])
                clip_reference_features = np.vstack([self.clip_features[r] for r in clip_reference_paths])
                
                # Calculate cosine similarity
                clip_similarity = cosine_similarity(clip_candidate_features, clip_reference_features)
                
                # Map to original indices
                for i, cand_path in enumerate(clip_candidate_paths):
                    ci = candidate_paths.index(cand_path)
                    for j, ref_path in enumerate(clip_reference_paths):
                        ri = self.reference_images.index(ref_path)
                        similarity_matrix[ci, ri] = clip_similarity[i, j]
        else:
            # BoVW similarity using histograms
            bovw_candidate_paths = [p for p in candidate_paths if p in self.image_histograms]
            bovw_reference_paths = [r for r in self.reference_images if r in self.image_histograms]
            
            if bovw_candidate_paths and bovw_reference_paths:
                # Extract histograms
                bovw_candidate_histograms = np.vstack([self.image_histograms[p] for p in bovw_candidate_paths])
                bovw_reference_histograms = np.vstack([self.image_histograms[r] for r in bovw_reference_paths])
                
                # Calculate histogram intersection or cosine similarity
                bovw_similarity = cosine_similarity(bovw_candidate_histograms, bovw_reference_histograms)
                
                # Map to original indices
                for i, cand_path in enumerate(bovw_candidate_paths):
                    ci = candidate_paths.index(cand_path)
                    for j, ref_path in enumerate(bovw_reference_paths):
                        ri = self.reference_images.index(ref_path)
                        similarity_matrix[ci, ri] = bovw_similarity[i, j]
        
        return similarity_matrix
    
    
    
    def select_next_view(self, candidate_names: List[str], n_select: int = 1) -> List[str]:
        """Select next viewpoint using features and source gaussian information
        
        1. Ensure sufficient overlap with existing views (at least min_overlap_ratio)
        2. Ensure viewpoint diversity (less than max_overlap_ratio)
        3. Prioritize views that can see unprocessed source gaussians
        
        Args:
            candidate_names: List of candidate image names
            n_select: Number of images to select

        Returns:
            List[str]: List of selected image names
        """
        # Convert to full paths
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # Abundant processing : process_images got this covered. keep it just in case.
        # # Process unprocessed images based on feature type
        # if self.feature_type == 'clip':
        #     # Process with CLIP
        #     for path in candidate_paths[:]:  # Use a copy for iteration while removing items
        #         if path not in self.clip_features:
        #             clip_features = self.extract_clip_features(path)
        #             if clip_features is not None:
        #                 self.clip_features[path] = clip_features
        #             else:
        #                 print(f"Failed to extract CLIP features from {path}, removing from candidates")
        #                 candidate_paths.remove(path)
        # else:
        #     # Process with traditional features
        #     for path in candidate_paths[:]:  # Use a copy for iteration while removing items
        #         if path not in self.image_histograms:
        #             if path not in self.image_features:
        #                 keypoints, descriptors = self.extract_features(path)
        #                 if descriptors is not None and len(descriptors) > 0:
        #                     self.image_features[path] = {
        #                         'keypoints': keypoints,
        #                         'descriptors': descriptors
        #                     }
                            
        #                     # Calculate histogram
        #                     if self.codebook is not None:
        #                         histogram = self.compute_image_histogram(descriptors)
        #                         self.image_histograms[path] = histogram
        #                     else:
        #                         print(f"No codebook available for {path}, removing from candidates")
        #                         candidate_paths.remove(path)
        #                 else:
        #                     print(f"No features found in {path}, removing from candidates")
        #                     candidate_paths.remove(path)
        #             else:
        #                 # Features exist but no histogram yet
        #                 descriptors = self.image_features[path]['descriptors']
        #                 if self.codebook is not None:
        #                     histogram = self.compute_image_histogram(descriptors)
        #                     self.image_histograms[path] = histogram
        #                 else:
        #                     print(f"No codebook available for {path}, removing from candidates")
        #                     candidate_paths.remove(path)
        
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
        
        # Calculate basic scores
        scores = np.zeros(len(valid_indices))
        
        # Base score: Similarity to existing views (moderate overlap is preferred)
        normalized_similarities = (max_similarities[valid_indices] - self.min_overlap_ratio) / (self.max_overlap_ratio - self.min_overlap_ratio)
        normalized_similarities = np.clip(normalized_similarities, 0, 1)
        
        # Highest score for moderate similarity (around 0.5)
        similarity_scores = 1.0 - 2.0 * np.abs(normalized_similarities - 0.5)
        scores += similarity_scores * 0.7  # Reduce weight to make room for source gaussian factor
        
        # Add source gaussian factor if available
        if self.source_gaussians_data is not None:
            # Calculate score based on how well a view might see unprocessed source gaussians
            source_scores = self._calculate_source_gaussian_scores(
                [candidate_paths[i] for i in valid_indices],
                similarity_matrix[valid_indices]
            )
            
            if source_scores is not None:
                # Normalize source scores
                if np.max(source_scores) > 0:
                    source_scores = source_scores / np.max(source_scores)
                    # Add to total scores with a weight
                    scores += source_scores * 0.3  # Adjust weight as needed
                    print("Added source gaussian scores to view selection criteria")
        
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
    
        
    def _calculate_source_gaussian_scores(self, candidate_paths: List[str], similarities: np.ndarray) -> np.ndarray:
        """Calculate scores for candidates based on potential to see unprocessed source gaussians
        
        Args:
            candidate_paths: Paths to candidate images
            similarities: Similarity matrix for these candidates
            
        Returns:
            np.ndarray: Source gaussian scores for each candidate
        """
        if self.source_gaussians_data is None:
            return None
            
        # Count unprocessed gaussians for each reference camera
        unprocessed_counts = {}
        for key, data in self.source_gaussians_data.items():
            if not key.startswith('source_gaussians') or 'indices' not in data:
                continue
            
            try:
                # Extract camera index from key (e.g., 'source_gaussians1_data' -> 0)
                cam_idx = int(key.replace('source_gaussians', '').replace('_data', '')) - 1
                
                # Count unprocessed gaussians
                if 'processed' in data:
                    unproc_count = np.sum(~data['processed'])
                else:
                    unproc_count = len(data['indices'])
                    
                if unproc_count > 0:
                    # Only consider cameras with unprocessed gaussians
                    unprocessed_counts[cam_idx] = unproc_count
            except (ValueError, IndexError):
                continue
        
        if not unprocessed_counts:
            return None
            
        # Debug info
        print("Unprocessed source gaussians by camera:")
        for cam_idx, count in unprocessed_counts.items():
            print(f"  Camera {cam_idx}: {count} unprocessed gaussians")
        
        # Check if we have reference images for these cameras
        ref_indices = []
        for cam_idx in unprocessed_counts.keys():
            # Find the reference image for this camera (if exists)
            if cam_idx < len(self.reference_images):
                ref_indices.append(cam_idx)
                
        if not ref_indices:
            return None
        
        # Calculate score based on similarity to cameras with unprocessed gaussians
        source_scores = np.zeros(len(candidate_paths))
        
        # Calculate weighted similarity to each camera with unprocessed gaussians
        for i, path in enumerate(candidate_paths):
            for cam_idx in ref_indices:
                # Skip if reference image isn't in similarity matrix
                if cam_idx >= similarities.shape[1]:
                    continue
                
                # Weight by number of unprocessed gaussians
                similarity = similarities[i, cam_idx]
                weight = unprocessed_counts.get(cam_idx, 0)
                
                # Add weighted similarity to score
                source_scores[i] += similarity * weight
                
        return source_scores
    
    def select_next_view_simple(self, candidate_names: List[str], n_select: int = 1) -> List[str]:
        """Select the images with highest similarity to existing views
        
        Args:
            candidate_names: List of candidate image names
            n_select: Number of images to select
            
        Returns:
            List[str]: List of selected image names
        """
        candidate_paths = [os.path.join(self.image_dir, name) for name in candidate_names]
        
        # Process unprocessed images based on feature type
        if self.feature_type == 'clip':
            # Process with CLIP
            for path in candidate_paths[:]:  # Use a copy for iteration while removing items
                if path not in self.clip_features:
                    clip_features = self.extract_clip_features(path)
                    if clip_features is not None:
                        self.clip_features[path] = clip_features
                    else:
                        print(f"Failed to extract CLIP features from {path}, removing from candidates")
                        candidate_paths.remove(path)
        else:
            # Process with traditional features
            for path in candidate_paths[:]:  # Use a copy for iteration while removing items
                if path not in self.image_histograms:
                    if path not in self.image_features:
                        keypoints, descriptors = self.extract_features(path)
                        if descriptors is not None and len(descriptors) > 0:
                            self.image_features[path] = {
                                'keypoints': keypoints,
                                'descriptors': descriptors
                            }
                            
                            # Calculate histogram
                            if self.codebook is not None:
                                histogram = self.compute_image_histogram(descriptors)
                                self.image_histograms[path] = histogram
                            else:
                                print(f"No codebook available for {path}, removing from candidates")
                                candidate_paths.remove(path)
                        else:
                            print(f"No features found in {path}, removing from candidates")
                            candidate_paths.remove(path)
                    else:
                        # Features exist but no histogram yet
                        descriptors = self.image_features[path]['descriptors']
                        if self.codebook is not None:
                            histogram = self.compute_image_histogram(descriptors)
                            self.image_histograms[path] = histogram
                        else:
                            print(f"No codebook available for {path}, removing from candidates")
                            candidate_paths.remove(path)
        
        if not candidate_paths:
            raise ValueError("No valid candidate images to select from")
        
        # Calculate similarity matrix
        similarity_matrix = self.compute_similarity_matrix(candidate_paths)
        
        # Calculate maximum similarity for each candidate
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Select based on highest similarity
        best_indices = np.argsort(-max_similarities)[:n_select]
        selected_paths = [candidate_paths[i] for i in best_indices]
        
        selected_names = [os.path.basename(path) for path in selected_paths]
        
        print(f"Selected {len(selected_names)} images based on highest similarity:")
        for i, name in enumerate(selected_names):
            idx = best_indices[i]
            print(f"  {name}: similarity={max_similarities[idx]:.3f}")
        
        return selected_names
    
    
    # 2DGSのパラメータを用いた類似度計算
    # def compute_gaussian_similarity(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians) -> float:
    #     """
    #     Compute similarity based on 2D Gaussian distribution characteristics
        
    #     Args:
    #         gaussians1: First set of 2D Gaussians
    #         gaussians2: Second set of 2D Gaussians
            
    #     Returns:
    #         float: Similarity score (0-1)
    #     """
    #     # 1. Similarity of center point spatial distributions
    #     means1 = gaussians1.means
    #     means2 = gaussians2.means
        
    #     if hasattr(means1, 'detach'):
    #         means1 = means1.detach().cpu().numpy()
    #     if hasattr(means2, 'detach'):
    #         means2 = means2.detach().cpu().numpy()
            
    #     # Compare distribution centers and variances
    #     center1 = np.mean(means1, axis=0)
    #     center2 = np.mean(means2, axis=0)
    #     center_dist = np.linalg.norm(center1 - center2) / np.linalg.norm(center1 + center2 + 1e-6)
        
    #     var1 = np.var(means1, axis=0)
    #     var2 = np.var(means2, axis=0)
    #     var_ratio = np.mean(np.maximum(var1, var2) / np.maximum(np.minimum(var1, var2), 1e-6))
    #     var_score = 1.0 / (1.0 + np.log(1 + var_ratio))
        
    #     # 2. Similarity of scale distributions
    #     scales1 = gaussians1.scales
    #     scales2 = gaussians2.scales
        
    #     if hasattr(scales1, 'detach'):
    #         scales1 = scales1.detach().cpu().numpy()
    #     if hasattr(scales2, 'detach'):
    #         scales2 = scales2.detach().cpu().numpy()
        
    #     scale1 = np.mean(scales1, axis=0)
    #     scale2 = np.mean(scales2, axis=0)
    #     scale_ratio = np.mean(np.maximum(scale1, scale2) / np.maximum(np.minimum(scale1, scale2), 1e-6))
    #     scale_score = 1.0 / (1.0 + np.log(1 + scale_ratio))
        
    #     # 3. Similarity of color distributions
    #     rgb1 = gaussians1.rgb
    #     rgb2 = gaussians2.rgb
        
    #     if hasattr(rgb1, 'detach'):
    #         rgb1 = rgb1.detach().cpu().numpy()
    #     if hasattr(rgb2, 'detach'):
    #         rgb2 = rgb2.detach().cpu().numpy()
        
    #     # Color histogram similarity (simplified implementation)
    #     hist1, _ = np.histogramdd(rgb1, bins=8, range=[[0, 1], [0, 1], [0, 1]])
    #     hist2, _ = np.histogramdd(rgb2, bins=8, range=[[0, 1], [0, 1], [0, 1]])
        
    #     hist1 = hist1 / np.sum(hist1)
    #     hist2 = hist2 / np.sum(hist2)
        
    #     color_sim = np.sum(np.minimum(hist1, hist2))
        
    #     # 4. Spatial diversity of Gaussian distributions
    #     diversity1 = np.sqrt(np.sum(var1))
    #     diversity2 = np.sqrt(np.sum(var2))
    #     diversity_ratio = min(diversity1, diversity2) / max(diversity1, diversity2)
        
    #     # Calculate final similarity score
    #     final_score = (
    #         0.2 * (1.0 - center_dist) +  # Center position similarity
    #         0.3 * var_score +            # Variance similarity
    #         0.2 * scale_score +          # Scale similarity
    #         0.2 * color_sim +            # Color distribution similarity
    #         0.1 * diversity_ratio        # Spatial diversity similarity
    #     )
        
    #     return final_score

    # def estimate_view_angle_change(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians) -> float:
    #     """
    #     Estimate view angle change from two Gaussian sets and convert to score
        
    #     Args:
    #         gaussians1: First set of 2D Gaussians
    #         gaussians2: Second set of 2D Gaussians
            
    #     Returns:
    #         float: View angle change score (0-1, higher means less change)
    #     """
    #     # Compare Gaussian distribution spread (covariance matrices)
    #     covs1 = gaussians1.covs
    #     covs2 = gaussians2.covs
        
    #     if hasattr(covs1, 'detach'):
    #         covs1 = covs1.detach().cpu().numpy()
    #     if hasattr(covs2, 'detach'):
    #         covs2 = covs2.detach().cpu().numpy()
        
    #     # Calculate average determinant (area) of covariance matrices
    #     det1 = np.mean([np.linalg.det(cov) for cov in covs1])
    #     det2 = np.mean([np.linalg.det(cov) for cov in covs2])
        
    #     # Determinant ratio (higher means larger view angle change)
    #     ratio = max(det1, det2) / max(min(det1, det2), 1e-10)
        
    #     # View angle change score (0-1, higher means less change)
    #     return 1.0 / (1.0 + np.log(1 + ratio))

    # def compute_initial_pair_score_gs(self, 
    #                             img1_name: str, 
    #                             img2_name: str, 
    #                             gaussians1: TwoDGaussians,
    #                             gaussians2: TwoDGaussians,
    #                             feature_sim: float) -> float:
    #     """
    #     Compute score for initial pair selection using features and 2D Gaussians
        
    #     Args:
    #         img1_name: First image name
    #         img2_name: Second image name
    #         gaussians1: First set of 2D Gaussians
    #         gaussians2: Second set of 2D Gaussians
    #         feature_sim: Feature similarity (from CLIP or BoVW)
            
    #     Returns:
    #         float: Final score
    #     """
    #     # 1. Gaussian-based similarity
    #     gauss_sim = self.compute_gaussian_similarity(gaussians1, gaussians2)
        
    #     # 2. View angle change score
    #     angle_score = self.estimate_view_angle_change(gaussians1, gaussians2)
        
    #     # 3. Gaussian count (richness) score
    #     count_ratio = min(gaussians1.k, gaussians2.k) / max(gaussians1.k, gaussians2.k)
        
    #     # Calculate final score
    #     final_score = (
    #         0.3 * feature_sim +   # Feature similarity (CLIP or BoVW)
    #         0.4 * gauss_sim +     # Gaussian distribution similarity
    #         0.2 * angle_score +   # Low view angle change
    #         0.1 * count_ratio     # Balanced Gaussian count
    #     )
        
    #     print(f"Pair {img1_name}-{img2_name}: feature_sim={feature_sim:.3f}, gauss_sim={gauss_sim:.3f}, "
    #         f"angle_score={angle_score:.3f}, count_ratio={count_ratio:.3f}, final={final_score:.3f}")
        
    #     return final_score