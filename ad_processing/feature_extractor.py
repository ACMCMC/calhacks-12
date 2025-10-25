"""
Feature Extractor for Project Aura
Extracts metadata features from ad creatives (images/videos).
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import PIL.Image
import io
from transformers import pipeline

class AdFeatureExtractor:
    """Extracts features and metadata from ad creatives."""

    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        # Initialize OCR (if available)
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'])
            self.has_ocr = True
        except ImportError:
            print("⚠ EasyOCR not available, OCR features disabled")
            self.has_ocr = False

    def extract_features(
        self,
        image_path: Optional[str] = None,
        image: Optional[PIL.Image.Image] = None,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive features from ad creative.

        Args:
            image_path: Path to image file
            image: PIL Image object
            text_content: Extracted text content (from OCR or provided)

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Load image
        if image_path and not image:
            image = PIL.Image.open(image_path).convert("RGB")
        elif not image:
            # No image provided
            return self._extract_text_only_features(text_content or "")

        # Convert to numpy array for OpenCV
        img_array = np.array(image)

        # Basic image features
        features.update(self._extract_image_features(img_array))

        # OCR text extraction
        if self.has_ocr:
            ocr_text = self._extract_text_ocr(img_array)
            if ocr_text:
                text_content = ocr_text
                features['ocr_text'] = ocr_text

        # Text-based features
        if text_content:
            features.update(self._extract_text_features(text_content))

        return features

    def _extract_image_features(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Extract basic image features."""
        features = {}

        # Color analysis
        features['dominant_color'] = self._get_dominant_color(img_array)

        # Brightness
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        features['brightness'] = float(np.mean(gray) / 255.0)

        # Contrast
        features['contrast'] = float(gray.std() / 255.0)

        # Motion/activity (for videos, but can work on images too)
        features['avg_motion'] = self._calculate_motion(gray)

        # Has call-to-action elements (simple heuristic)
        features['has_cta'] = self._detect_cta(img_array)

        # Image dimensions
        h, w = img_array.shape[:2]
        features['aspect_ratio'] = float(w / h)
        features['resolution'] = f"{w}x{h}"

        return features

    def _extract_text_ocr(self, img_array: np.ndarray) -> Optional[str]:
        """Extract text using OCR."""
        if not self.has_ocr:
            return None

        try:
            results = self.reader.readtext(img_array)
            text = " ".join([result[1] for result in results if result[2] > 0.5])
            return text.strip() if text else None
        except Exception as e:
            print(f"OCR failed: {e}")
            return None

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text content."""
        features = {}

        # Sentiment analysis
        try:
            sentiment_result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            features['sentiment'] = sentiment_result['label'].lower()
            features['sentiment_score'] = float(sentiment_result['score'])
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            features['sentiment'] = 'neutral'
            features['sentiment_score'] = 0.5

        # Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Hashtags, mentions, etc.
        features['has_hashtag'] = '#' in text
        features['has_mention'] = '@' in text
        features['has_url'] = 'http' in text.lower()

        # Keywords extraction (simple)
        features['keywords'] = self._extract_keywords(text)

        return features

    def _extract_text_only_features(self, text: str) -> Dict[str, Any]:
        """Extract features when only text is available."""
        features = self._extract_text_features(text)
        features.update({
            'has_image': False,
            'dominant_color': 'unknown',
            'brightness': 0.5,
            'contrast': 0.5,
            'avg_motion': 0.0,
            'has_cta': False,
            'aspect_ratio': 1.0,
            'resolution': 'text_only'
        })
        return features

    def _get_dominant_color(self, img_array: np.ndarray) -> str:
        """Get dominant color using k-means clustering."""
        try:
            pixels = img_array.reshape(-1, 3)
            pixels = pixels[np.random.choice(pixels.shape[0], min(1000, pixels.shape[0]), replace=False)]

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, n_init=10)
            kmeans.fit(pixels)

            dominant_color = kmeans.cluster_centers_[0].astype(int)
            return f"rgb({dominant_color[0]},{dominant_color[1]},{dominant_color[2]})"
        except:
            return "unknown"

    def _calculate_motion(self, gray_img: np.ndarray) -> float:
        """Calculate average motion/activity in image."""
        try:
            # Simple motion detection using gradient magnitude
            sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            return float(np.mean(magnitude) / 255.0)
        except:
            return 0.0

    def _detect_cta(self, img_array: np.ndarray) -> bool:
        """Simple CTA detection heuristic."""
        # Look for bright, high-contrast areas (common in CTAs)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Check if there's a significant bright area
        bright_ratio = np.sum(thresh > 0) / thresh.size
        return bright_ratio > 0.05  # More than 5% bright pixels

    def _extract_keywords(self, text: str) -> list:
        """Simple keyword extraction."""
        # Remove common stop words and get meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        keywords = [word.strip('.,!?') for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:10]  # Return up to 10 unique keywords

# Convenience function
def extract_ad_features(image_path=None, image=None, text_content=None) -> Dict[str, Any]:
    """Convenience function to extract features from an ad."""
    extractor = AdFeatureExtractor()
    return extractor.extract_features(
        image_path=image_path,
        image=image,
        text_content=text_content
    )

if __name__ == "__main__":
    # Test feature extraction
    extractor = AdFeatureExtractor()

    # Test with text only
    features = extractor.extract_features(text_content="Buy now! Amazing deals on luxury watches!")
    print("Text-only features:", features)

    # Test with dummy image features (would need actual image)
    print("Feature extraction module ready")