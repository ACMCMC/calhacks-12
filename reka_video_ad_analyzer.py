"""
Reka Video Ad Intelligence Analyzer
Uses Reka API for comprehensive video advertisement analysis
"""

import requests
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time
import os

from dotenv import load_dotenv
load_dotenv()

reka_key = os.getenv("REKA_API_KEY")

@dataclass
class VideoAnalysisResult:
    """Container for video analysis results"""
    video_path: str
    status: str
    analysis: str
    features: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RekaVideoAnalyzer:
    """
    Video advertisement analyzer using Reka API
    Supports multimodal analysis of video content
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Reka video analyzer
        
        Args:
            api_key: Reka API key from https://platform.reka.ai
        """
        self.api_key = api_key
        self.base_url = "https://api.reka.ai/v1/chat"
        self.headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def _encode_video(self, video_path: str) -> str:
        """
        Encode video file to base64
        
        Args:
            video_path: Path to video file
            
        Returns:
            Base64 encoded video string
        """
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            return base64.b64encode(video_bytes).decode('utf-8')
    
    def _get_video_mime_type(self, video_path: str) -> str:
        """
        Determine MIME type from file extension
        
        Args:
            video_path: Path to video file
            
        Returns:
            MIME type string
        """
        extension = Path(video_path).suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska'
        }
        return mime_types.get(extension, 'video/mp4')
    
    def _create_comprehensive_prompt(self) -> str:
        """
        Create comprehensive video analysis prompt
        """
        return """You are an expert in video advertisement analysis. Analyze this video advertisement comprehensively and provide detailed insights.

**SECTION 1: FOUNDATIONAL VISUAL FEATURES**
1. Object and Brand Recognition:
   - List all major objects visible throughout the video
   - Identify any brand logos, product packaging, or brand elements
   - Determine the primary product category (automotive, fashion, food, tech, etc.)
   - Count distinct people, products, and key objects

2. Text Content (OCR):
   - Extract ALL visible text throughout the video (on-screen text, captions, product labels)
   - Identify call-to-action phrases (Shop Now, Learn More, Buy Now, etc.)
   - Note promotional offers or pricing information
   - Assess text readability and prominence

3. Content Categorization:
   - Ad format type (product demo, lifestyle, testimonial, comparison, brand awareness)
   - Target audience indicators
   - Production quality (low/medium/high/premium)

**SECTION 2: VISUAL SENTIMENT AND CREATIVE EXECUTION**
1. Visual Sentiment:
   - Overall emotional tone (happy, serious, urgent, calm, luxurious, playful)
   - If people are present: describe their expressions and emotions
   - Scene-based mood (lighting, color psychology, atmosphere)
   - Intended viewer emotional response

2. Aesthetic Quality:
   - Dominant color palette and color harmony
   - Visual complexity (simple/moderate/complex)
   - Compositional balance and quality
   - Overall aesthetic score (1-10)

**SECTION 3: TEMPORAL DYNAMICS AND PACING**
1. Video Pacing:
   - Estimate number of scene cuts/transitions
   - Pacing style (fast-paced with quick cuts vs slow and deliberate)
   - Motion intensity (static shots vs high-action dynamic content)

2. Shot Types:
   - Identify primary shot types used (close-ups, medium shots, wide shots)
   - Camera movement (static, panning, tracking, etc.)

**SECTION 4: AUDIO ANALYSIS** (if audio is present)
1. Speech Content:
   - Transcribe or summarize any spoken dialogue or voiceover
   - Identify key marketing messages and claims
   - Note tone of voice and speaking style

2. Audio Elements:
   - Presence of background music (type, mood, tempo)
   - Sound effects and their role
   - Overall audio-visual congruence

**SECTION 5: NARRATIVE AND STORYTELLING**
1. Story Arc:
   - Beginning: How does the ad open/hook the viewer?
   - Middle: What is the main message or demonstration?
   - End: How does it conclude? (product reveal, CTA, tagline)

2. Action Sequence:
   - Key actions performed in the video
   - Product interaction moments
   - Narrative progression

**SECTION 6: CREATIVE STRATEGY**
1. Persuasion Strategy:
   - Primary appeal (emotional, logical, urgency, social proof, authority)
   - Unique selling proposition
   - Brand positioning

2. Innovation and Creativity:
   - Creative elements that stand out
   - Unusual or memorable techniques
   - Innovation score (1-10)

Provide a comprehensive analysis addressing all these sections with specific observations and quantitative assessments where possible."""
    
    def analyze_video(
        self, 
        video_path: str, 
        custom_question: Optional[str] = None,
        model: str = "reka-core-20240501"
    ) -> VideoAnalysisResult:
        """
        Analyze a video advertisement
        
        Args:
            video_path: Path to the video file
            custom_question: Optional custom question about the video
            model: Reka model to use (reka-core, reka-flash, reka-edge)
            
        Returns:
            VideoAnalysisResult object
        """
        try:
            print(f"Analyzing video: {video_path}")
            print("Encoding video... (this may take a moment for large files)")
            
            # Encode video
            video_base64 = self._encode_video(video_path)
            mime_type = self._get_video_mime_type(video_path)
            
            print(f"Video encoded ({len(video_base64)} bytes)")
            print(f"MIME type: {mime_type}")
            
            # Prepare the prompt
            prompt = custom_question if custom_question else self._create_comprehensive_prompt()
            
            # Prepare request payload
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": f"data:{mime_type};base64,{video_base64}"
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "model": model,
                "temperature": 0.3,
                "max_tokens": 4096
            }
            
            print(f"Sending request to Reka API (model: {model})...")
            print("This may take 30-60 seconds for video analysis...")
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['responses'][0]['message']['content']
                
                return VideoAnalysisResult(
                    video_path=video_path,
                    status='success',
                    analysis=analysis_text
                )
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                return VideoAnalysisResult(
                    video_path=video_path,
                    status='error',
                    analysis='',
                    error=error_msg
                )
                
        except Exception as e:
            return VideoAnalysisResult(
                video_path=video_path,
                status='error',
                analysis='',
                error=str(e)
            )
    
    def ask_question(
        self, 
        video_path: str, 
        question: str,
        model: str = "reka-core-20240501"
    ) -> VideoAnalysisResult:
        """
        Ask a specific question about a video
        
        Args:
            video_path: Path to the video file
            question: Your question about the video
            model: Reka model to use
            
        Returns:
            VideoAnalysisResult object
        """
        return self.analyze_video(video_path, custom_question=question, model=model)
    
    def analyze_batch(
        self, 
        video_paths: List[str],
        model: str = "reka-core-20240501",
        delay: float = 2.0
    ) -> List[VideoAnalysisResult]:
        """
        Analyze multiple videos
        
        Args:
            video_paths: List of video file paths
            model: Reka model to use
            delay: Delay between requests (seconds) to avoid rate limits
            
        Returns:
            List of VideoAnalysisResult objects
        """
        results = []
        for i, path in enumerate(video_paths, 1):
            print(f"\n{'='*70}")
            print(f"Processing video {i}/{len(video_paths)}")
            print(f"{'='*70}")
            
            result = self.analyze_video(path, model=model)
            results.append(result)
            
            # Add delay between requests
            if i < len(video_paths):
                print(f"\nWaiting {delay} seconds before next video...")
                time.sleep(delay)
        
        return results
    
    def extract_structured_features(
        self, 
        video_path: str,
        model: str = "reka-core-20240501"
    ) -> Dict[str, Any]:
        """
        Extract structured features from video
        
        Args:
            video_path: Path to video file
            model: Reka model to use
            
        Returns:
            Dictionary of structured features
        """
        structured_prompt = """Analyze this video advertisement and return a JSON object with the following structure:

{
  "visual_features": {
    "objects": ["list of detected objects"],
    "brands": ["detected brands/logos"],
    "scene_classification": ["list of scene classifications (indoor, outdoor, urban, nature, office, etc)"],
    "person detection": ["estimated age groups, gender, ethnicity, etc"],
    "category": "product category",
    "has_faces": true/false,
    "people_count": number,
    "cultural_cues": "holiday, language, icons, etc",
    "safety_compliance_flags": "NSFW/political/restricted"
  },
  "text_content": {
    "all_text": ["list of all visible text"],
    "cta_phrases": ["call-to-action phrases"],
    "brand_tagline": "tagline if visible"
  },
  "temporal_features": {
    "scene_count": "estimated number",
    "pacing": "fast/medium/slow",
    "average_shot_duration": "estimated in seconds",
    "motion_intensity": "low/medium/high"
  },
  "audio_features": {
    "has_voiceover": true/false,
    "has_music": true/false,
    "key_spoken_messages": ["main verbal messages"],
    "music_mood": "description"
  },
  "sentiment": {
    "overall_emotion": "primary emotion",
    "emotional_valence": "positive/negative/neutral",
    "mood": "description"
  },
  "creative_strategy": {
    "ad_format": "type",
    "production_quality": "low/medium/high/premium",
    "production_categorization": "photo, illustration, CGI, inforgraphic, meme, etc",
    "style": "minimalistic/luxury/festive/casual",
    "target_audience": "description split into keywords",
    "innovation_score": 1-10
  }
}

Return ONLY valid JSON, no additional text."""

        result = self.analyze_video(video_path, custom_question=structured_prompt, model=model)
        
        if result.status == 'success':
            try:
                # Try to parse JSON from response
                json_text = result.analysis.strip()
                if json_text.startswith('```'):
                    lines = json_text.split('\n')
                    json_text = '\n'.join(lines[1:-1])
                
                features = json.loads(json_text)
                return features
            except json.JSONDecodeError:
                return {'error': 'Failed to parse JSON', 'raw': result.analysis}
        else:
            return {'error': result.error}
    
    def print_analysis(self, result: VideoAnalysisResult):
        """
        Print formatted analysis results
        
        Args:
            result: VideoAnalysisResult object
        """
        print("\n" + "="*70)
        print("VIDEO ANALYSIS RESULTS")
        print("="*70)
        print(f"\n📹 VIDEO: {result.video_path}")
        print(f"STATUS: {result.status.upper()}")
        
        if result.status == 'success':
            print(f"\n{result.analysis}")
        else:
            print(f"\n❌ ERROR: {result.error}")
    
    def export_results(self, result: VideoAnalysisResult, output_path: str):
        """
        Export analysis results to JSON file
        
        Args:
            result: VideoAnalysisResult object
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"\n✅ Results exported to: {output_path}")


def main():
    analyzer = RekaVideoAnalyzer(api_key=reka_key)
        
    # Example 1: Comprehensive analysis
    print("\n\nEXAMPLE 1: COMPREHENSIVE VIDEO ANALYSIS")
    print("-"*70)
    
    video_path = "data/ads/videos/v0001.mp4"
    
    result = analyzer.analyze_video(video_path)
    analyzer.print_analysis(result)
    analyzer.export_results(result, "video_analysis.json")
    
    # Example 3: Extract structured features
    print("\n\n" + "="*70)
    print("EXAMPLE 3: EXTRACT STRUCTURED FEATURES")
    print("-"*70)
    
    #Uncomment to run
    features = analyzer.extract_structured_features(video_path)
    print(json.dumps(features, indent=2))

if __name__ == "__main__":
    main()