"""
Ad Processing Pipeline for PrivAds
Processes ad creatives to extract features, generate embeddings, and create metadata.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad_processing.encoder import AdEncoder
from ad_processing.feature_extractor import AdFeatureExtractor
from ad_processing.tagger import AdTagger

def process_ad_creative(
    ad_id: str,
    image_path: Optional[str] = None,
    text_content: Optional[str] = None,
    encoder: AdEncoder = None,
    feature_extractor: AdFeatureExtractor = None,
    tagger: AdTagger = None
) -> Dict[str, Any]:
    """
    Process a single ad creative and extract all features.

    Args:
        ad_id: Unique identifier for the ad
        image_path: Path to ad image/video file
        text_content: Optional text content
        encoder: Jina CLIP encoder instance
        feature_extractor: Feature extractor instance
        tagger: Tag generator instance

    Returns:
        Dictionary containing all processed data
    """
    result = {
        'ad_id': ad_id,
        'image_path': image_path,
        'text_content': text_content
    }

    # Encode to embedding
    if encoder:
        try:
            embedding = encoder.encode(text=text_content, image_path=image_path)
            result['raw_embedding'] = embedding.tolist()
        except Exception as e:
            print(f"⚠ Failed to encode ad {ad_id}: {e}")
            result['raw_embedding'] = None

    # Extract features
    if feature_extractor:
        try:
            features = feature_extractor.extract_features(
                image_path=image_path,
                text_content=text_content
            )
            result['features'] = features
        except Exception as e:
            print(f"⚠ Failed to extract features for ad {ad_id}: {e}")
            result['features'] = {}

    # Generate tags
    if tagger and 'features' in result:
        try:
            tags_data = tagger.generate_tags(
                features=result['features'],
                text_content=text_content or ""
            )
            result['tags'] = tags_data
        except Exception as e:
            print(f"⚠ Failed to generate tags for ad {ad_id}: {e}")
            result['tags'] = {'tags': [], 'type': 'general'}

    return result

def process_ad_directory(
    ad_directory: str,
    output_file: str = "ad_metadata.jsonl",
    text_descriptions: Optional[Dict[str, str]] = None
) -> None:
    """
    Process all ad creatives in a directory.

    Args:
        ad_directory: Path to directory containing ad files
        output_file: Output JSONL file path
        text_descriptions: Optional dict mapping filenames to text descriptions
    """
    ad_dir = Path(ad_directory)
    if not ad_dir.exists():
        raise FileNotFoundError(f"Ad directory not found: {ad_directory}")

    # Initialize processors
    print("Initializing processors...")
    encoder = AdEncoder()
    feature_extractor = AdFeatureExtractor()
    tagger = AdTagger()

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    ad_files = []
    for ext in image_extensions:
        ad_files.extend(ad_dir.glob(ext))

    if not ad_files:
        print("⚠ No image files found in ad_creatives")
        # Create some dummy ads for testing
        ad_files = [f"dummy_ad_{i}.jpg" for i in range(10)]

    print(f"Found {len(ad_files)} ad files to process")

    # Process each ad
    processed_ads = []
    for i, ad_file in enumerate(ad_files):
        ad_id = f"ad{i:04d}"

        # Convert to Path object if it's a string
        if isinstance(ad_file, str):
            ad_file_path = Path(ad_directory) / ad_file
        else:
            ad_file_path = ad_file

        # Get text description if available
        text_content = None
        if text_descriptions and str(ad_file_path) in text_descriptions:
            text_content = text_descriptions[str(ad_file_path)]
        elif ad_file_path.exists():  # Real file
            # Use filename as basic description
            text_content = ad_file_path.stem.replace('_', ' ').title()

        print(f"Processing {ad_id}: {ad_file_path}")

        result = process_ad_creative(
            ad_id=ad_id,
            image_path=str(ad_file_path) if ad_file_path.exists() else None,
            text_content=text_content,
            encoder=encoder,
            feature_extractor=feature_extractor,
            tagger=tagger
        )

        processed_ads.append(result)

    print(f"Debug: Processed {len(processed_ads)} ads total")

def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert other objects to string
        return str(obj)

    # Save to JSONL file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Debug: About to save {len(processed_ads)} ads to {output_path}")

    with open(output_path, 'w') as f:
        for i, ad in enumerate(processed_ads):
            print(f"Debug: Processing ad {i}")
            serializable_ad = make_json_serializable(ad)
            json.dump(serializable_ad, f)
            f.write('\n')

    print(f"✓ Processed {len(processed_ads)} ads")
    print(f"✓ Saved results to {output_path}")

    # Print summary
    tag_counts = {}
    type_counts = {}
    for ad in processed_ads:
        if 'tags' in ad:
            for tag in ad['tags'].get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            ad_type = ad['tags'].get('type', 'unknown')
            type_counts[ad_type] = type_counts.get(ad_type, 0) + 1

    print("\nTag Summary:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag}: {count}")

    print("\nType Summary:")
    for ad_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ad_type}: {count}")

def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("PrivAds: Ad Processing Pipeline")
    print("=" * 60)

    # Default paths
    ad_directory = "ad_creatives"  # Directory containing ad images
    output_file = "backend/data/ad_metadata.jsonl"

    # Create sample text descriptions for demo ads
    sample_descriptions = {
        "dummy_ad_0.jpg": "Sustainable children's clothing, eco-friendly materials",
        "dummy_ad_1.jpg": "Luxury watch, premium quality, expensive timepiece",
        "dummy_ad_2.jpg": "Affordable kids toys, colorful and fun educational games",
        "dummy_ad_3.jpg": "Organic food delivery, healthy meals for busy families",
        "dummy_ad_4.jpg": "Gaming laptop, high performance for gamers and professionals",
        "dummy_ad_5.jpg": "Vote for America, make America great again, conservative values",
        "dummy_ad_6.jpg": "Chess tournament, strategic thinking, competitive play",
        "dummy_ad_7.jpg": "Eco-friendly products, save the planet, sustainable living",
        "dummy_ad_8.jpg": "Luxury car, premium automotive excellence",
        "dummy_ad_9.jpg": "Health and wellness, fitness for life"
    }

    try:
        process_ad_directory(
            ad_directory=ad_directory,
            output_file=output_file,
            text_descriptions=sample_descriptions
        )

        print("\n" + "=" * 60)
        print("✅ Ad Processing Pipeline Complete!")
        print("=" * 60)
        print(f"📁 Results saved to: {output_file}")
        print("\nNext steps:")
        print("  - Load data into databases: python pipeline/load_databases.py")
        print("  - Start backend API: cd backend && python main.py")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()