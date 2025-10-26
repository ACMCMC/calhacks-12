"""
Complete Ad Intelligence Feature Extraction Pipeline
Implements the state-of-the-art methodology from the technical survey
"""

import google.generativeai as genai
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict
import PIL.Image

import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class VisualFeatures:
    """Visual feature extraction results"""
    objects: List[str]
    object_counts: Dict[str, int]
    brands: List[str]
    category: str
    has_faces: bool
    face_count: int
    
@dataclass
class TextFeatures:
    """Text extraction results"""
    extracted_text: List[str]
    full_text: str
    has_cta: bool
    cta_phrases: List[str]
    text_prominence: str  # low/medium/high
    
@dataclass
class AestheticFeatures:
    """Aesthetic quality features"""
    dominant_colors: List[str]
    color_harmony: str
    color_temperature: str  # warm/cool/neutral
    visual_complexity: int  # 1-10
    compositional_balance: int  # 1-10
    rule_of_thirds_score: int  # 1-10
    overall_aesthetic_score: float  # 1-10
    
@dataclass
class SentimentFeatures:
    """Emotional and sentiment features"""
    overall_emotion: str
    emotional_valence: str  # positive/negative/neutral
    expressed_emotions: List[str]
    scene_mood: str
    emotional_intensity: int  # 1-10
    
@dataclass
class CreativeFeatures:
    """Creative strategy features"""
    ad_format: str
    shot_type: str
    production_quality: str  # low/medium/high/premium
    target_audience: str
    persuasion_strategy: str
    brand_prominence: str  # subtle/moderate/dominant
    innovation_score: int  # 1-10
    
@dataclass
class ComputationalMetrics:
    """Computational aesthetic metrics"""
    visual_complexity_score: int  # 1-10
    color_entropy: int  # 1-10
    clutter_level: int  # 1-10
    text_to_image_ratio: float  # percentage
    compositional_quality: int  # 1-10


class AdIntelligenceExtractor:
    """
    Complete feature extraction system for ad intelligence analysis
    Uses Google Gemini API for comprehensive multi-modal analysis
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the feature extractor
        
        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        for model in genai.list_models():
            print(model.name)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def _create_comprehensive_prompt(self) -> str:
        """
        Create detailed extraction prompt based on technical survey
        """
        return """You are an expert ad intelligence system performing state-of-the-art feature extraction.

Analyze this advertisement image and extract features in the following structured JSON format:

{
  "visual_features": {
    "objects": ["list of all detected objects"],
    "object_counts": {"object_type": count},
    "brands": ["detected brand names/logos"],
    "category": "primary product category",
    "has_faces": true/false,
    "face_count": number
  },
  "text_features": {
    "extracted_text": ["all visible text strings"],
    "full_text": "concatenated text",
    "has_cta": true/false,
    "cta_phrases": ["Shop Now", "Learn More", etc],
    "text_prominence": "low/medium/high"
  },
  "aesthetic_features": {
    "dominant_colors": ["color1", "color2", "color3"],
    "color_harmony": "complementary/analogous/monochromatic/triadic",
    "color_temperature": "warm/cool/neutral",
    "visual_complexity": "1-10 integer",
    "compositional_balance": "1-10 integer",
    "rule_of_thirds_score": "1-10 integer",
    "overall_aesthetic_score": "1-10 float"
  },
  "sentiment_features": {
    "overall_emotion": "primary emotion",
    "emotional_valence": "positive/negative/neutral",
    "expressed_emotions": ["happy", "excited", etc],
    "scene_mood": "description of mood",
    "emotional_intensity": "1-10 integer"
  },
  "creative_features": {
    "ad_format": "product shot/lifestyle/testimonial/comparison",
    "shot_type": "close-up/medium/wide/overhead",
    "production_quality": "low/medium/high/premium",
    "target_audience": "demographic description",
    "persuasion_strategy": "emotional/logical/urgency/social proof",
    "brand_prominence": "subtle/moderate/dominant",
    "innovation_score": "1-10 integer"
  },
  "computational_metrics": {
    "visual_complexity_score": "1-10 integer",
    "color_entropy": "1-10 integer",
    "clutter_level": "1-10 integer",
    "text_to_image_ratio": "0.0-100.0 float",
    "compositional_quality": "1-10 integer"
  }
}

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON, no additional text or markdown
2. All numeric scores should be integers 1-10 unless specified otherwise
3. Be precise and specific in all classifications
4. Extract ALL visible text completely
5. Identify specific brand names if visible
6. Provide detailed object detection (people, products, vehicles, etc)

Analyze the image now and return the JSON structure:"""

    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all features from an advertisement image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            img = PIL.Image.open(image_path)
            prompt = self._create_comprehensive_prompt()
            
            print(f"Extracting features from: {image_path}")
            print("Processing... (this may take 15-30 seconds)")
            
            response = self.model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower for more consistent structured output
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=4096,
                )
            )
            
            # Parse JSON response
            json_text = response.text.strip()
            
            # Clean markdown if present
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1])
            
            features = json.loads(json_text)
            
            # Convert to dataclass objects
            result = {
                'image_path': image_path,
                'status': 'success',
                'visual': VisualFeatures(**features['visual_features']),
                'text': TextFeatures(**features['text_features']),
                'aesthetic': AestheticFeatures(**features['aesthetic_features']),
                'sentiment': SentimentFeatures(**features['sentiment_features']),
                'creative': CreativeFeatures(**features['creative_features']),
                'metrics': ComputationalMetrics(**features['computational_metrics']),
                'raw_json': features
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response.text[:500]}")
            return {
                'image_path': image_path,
                'status': 'error',
                'error': f'JSON parsing failed: {str(e)}',
                'raw_response': response.text
            }
        except Exception as e:
            return {
                'image_path': image_path,
                'status': 'error',
                'error': str(e)
            }
    
    def extract_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of feature extraction results
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n{'='*70}")
            print(f"Processing image {i}/{len(image_paths)}")
            print(f"{'='*70}")
            result = self.extract_features(path)
            results.append(result)
        return results
    
    def generate_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert extracted features to numeric feature vector
        
        Args:
            features: Extracted features dictionary
            
        Returns:
            Numpy array of numeric features
        """
        if features['status'] != 'success':
            return None
            
        vector = []
        
        # Object counts
        vector.append(features['visual'].face_count)
        vector.append(len(features['visual'].objects))
        vector.append(len(features['visual'].brands))
        
        # Text features
        vector.append(1 if features['text'].has_cta else 0)
        vector.append(len(features['text'].extracted_text))
        
        # Aesthetic scores
        vector.append(features['aesthetic'].visual_complexity)
        vector.append(features['aesthetic'].compositional_balance)
        vector.append(features['aesthetic'].rule_of_thirds_score)
        vector.append(features['aesthetic'].overall_aesthetic_score)
        
        # Sentiment
        vector.append(features['sentiment'].emotional_intensity)
        
        # Creative scores
        vector.append(features['creative'].innovation_score)
        
        # Computational metrics
        vector.append(features['metrics'].visual_complexity_score)
        vector.append(features['metrics'].color_entropy)
        vector.append(features['metrics'].clutter_level)
        vector.append(features['metrics'].text_to_image_ratio)
        vector.append(features['metrics'].compositional_quality)
        
        return np.array(vector)
    
    def print_feature_summary(self, features: Dict[str, Any]):
        """
        Print a human-readable summary of extracted features
        
        Args:
            features: Extracted features dictionary
        """
        if features['status'] != 'success':
            print(f"Error: {features['error']}")
            return
        
        print("\n" + "="*70)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*70)
        
        print(f"\n📸 IMAGE: {features['image_path']}")
        
        print(f"\n🎯 VISUAL FEATURES:")
        print(f"  Category: {features['visual'].category}")
        print(f"  Objects: {', '.join(features['visual'].objects)}")
        print(f"  Brands: {', '.join(features['visual'].brands) if features['visual'].brands else 'None detected'}")
        print(f"  Faces: {features['visual'].face_count}")
        
        print(f"\n📝 TEXT FEATURES:")
        print(f"  Full Text: {features['text'].full_text[:100]}...")
        print(f"  Has CTA: {features['text'].has_cta}")
        print(f"  CTA Phrases: {', '.join(features['text'].cta_phrases)}")
        
        print(f"\n🎨 AESTHETIC FEATURES:")
        print(f"  Dominant Colors: {', '.join(features['aesthetic'].dominant_colors)}")
        print(f"  Color Harmony: {features['aesthetic'].color_harmony}")
        print(f"  Overall Aesthetic Score: {features['aesthetic'].overall_aesthetic_score}/10")
        
        print(f"\n😊 SENTIMENT FEATURES:")
        print(f"  Overall Emotion: {features['sentiment'].overall_emotion}")
        print(f"  Emotional Valence: {features['sentiment'].emotional_valence}")
        print(f"  Emotional Intensity: {features['sentiment'].emotional_intensity}/10")
        
        print(f"\n💡 CREATIVE FEATURES:")
        print(f"  Ad Format: {features['creative'].ad_format}")
        print(f"  Production Quality: {features['creative'].production_quality}")
        print(f"  Target Audience: {features['creative'].target_audience}")
        print(f"  Innovation Score: {features['creative'].innovation_score}/10")
        
        print(f"\n📊 COMPUTATIONAL METRICS:")
        print(f"  Visual Complexity: {features['metrics'].visual_complexity_score}/10")
        print(f"  Color Entropy: {features['metrics'].color_entropy}/10")
        print(f"  Clutter Level: {features['metrics'].clutter_level}/10")
        print(f"  Text Coverage: {features['metrics'].text_to_image_ratio:.1f}%")
        
        # Generate feature vector
        vector = self.generate_feature_vector(features)
        print(f"\n🔢 FEATURE VECTOR (16 dimensions):")
        print(f"  {vector}")
        
    def export_features(self, features: Dict[str, Any], output_path: str):
        """
        Export features to JSON file
        
        Args:
            features: Extracted features
            output_path: Output file path
        """
        if features['status'] != 'success':
            print(f"Cannot export - extraction failed: {features['error']}")
            return
        
        # Convert dataclasses to dicts
        export_data = {
            'image_path': features['image_path'],
            'visual': asdict(features['visual']),
            'text': asdict(features['text']),
            'aesthetic': asdict(features['aesthetic']),
            'sentiment': asdict(features['sentiment']),
            'creative': asdict(features['creative']),
            'metrics': asdict(features['metrics'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Features exported to: {output_path}")


def main():
    """
    Example usage demonstrating the complete pipeline
    """
    # Configuration
    gem_key = os.getenv('GEM_KEY')
    
    # Initialize extractor
    extractor = AdIntelligenceExtractor(api_key=gem_key)
    
    print("="*70)
    print("AD INTELLIGENCE FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    # Example 1: Single image analysis
    image_path = "data/images/image_0.jpeg"
    
    features = extractor.extract_features(image_path)
    
    if features['status'] == 'success':
        # Print summary
        extractor.print_feature_summary(features)
        
        # Export to JSON
        extractor.export_features(features, "ad_features.json")
        
    else:
        print(f"\n❌ Extraction failed: {features['error']}")
    
    
    # Example 2: Batch processing
    print("\n\n" + "="*70)
    print("BATCH PROCESSING EXAMPLE")
    print("="*70)
    
    image_paths = [
        "data/images/image_0.jpeg",
        "data/images/image_1.jpeg",
        "data/images/image_2.jpeg",
        "data/images/image_3.jpeg",
        "data/images/image_4.jpeg",
        "data/images/image_5.jpeg",
        "data/images/image_6.jpeg",
        "data/images/image_7.jpeg",
        "data/images/image_8.jpeg",
        "data/images/image_9.jpeg",
    ]
    
    # Uncomment to run batch processing
    # results = extractor.extract_batch(image_paths)
    # 
    # # Generate feature matrix
    # feature_matrix = []
    # for result in results:
    #     if result['status'] == 'success':
    #         vector = extractor.generate_feature_vector(result)
    #         feature_matrix.append(vector)
    # 
    # feature_matrix = np.array(feature_matrix)
    # print(f"\nFeature matrix shape: {feature_matrix.shape}")
    # print(f"Features per ad: {feature_matrix.shape[1]}")
    
    
    # Example 3: Custom analysis
    print("\n\n" + "="*70)
    print("CUSTOM ANALYSIS CAPABILITIES")
    print("="*70)
    print("""
The extractor provides:

1. VISUAL FEATURES
   - Object detection (YOLOv8-style)
   - Brand/logo recognition
   - Face detection and counting
   - Product category classification

2. TEXT FEATURES  
   - OCR extraction (EasyOCR/PaddleOCR-style)
   - Call-to-action detection
   - Text prominence analysis
   
3. AESTHETIC FEATURES
   - Color analysis (dominant colors, harmony, temperature)
   - Visual complexity metrics
   - Compositional analysis (Rule of Thirds)
   - Overall aesthetic quality score

4. SENTIMENT FEATURES
   - Emotional tone detection
   - Valence classification
   - Scene mood analysis
   
5. CREATIVE STRATEGY
   - Ad format classification
   - Production quality assessment
   - Target audience identification
   - Persuasion strategy analysis
   
6. COMPUTATIONAL METRICS
   - Visual complexity quantification
   - Color entropy
   - Clutter assessment
   - Text coverage ratio

All features are exported as:
- Structured dataclasses
- JSON format
- Numeric feature vectors (for ML models)
    """)


if __name__ == "__main__":
    main()
