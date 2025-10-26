"""
Database Loading Pipeline for PrivAds
Loads processed ad data into Chroma (vectors) and Elasticsearch (metadata).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import chromadb
from elasticsearch import Elasticsearch
import os

def load_chroma_database(
    metadata_file: str,
    chroma_path: str = "backend/chroma_db",
    collection_name: str = "ads"
) -> None:
    """
    Load ad embeddings and metadata into Chroma vector database.

    Args:
        metadata_file: Path to JSONL file with processed ad data
        chroma_path: Path to Chroma database
        collection_name: Name of Chroma collection
    """
    print("Loading data into Chroma...")

    # Initialize Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)

    # Read and process metadata file
    loaded_count = 0
    with open(metadata_file, 'r') as f:
        for line in f:
            ad_data = json.loads(line.strip())

            ad_id = ad_data['ad_id']
            embedding = ad_data.get('raw_embedding')

            if embedding is None:
                print(f"⚠ Skipping ad {ad_id}: no embedding")
                continue

            # Prepare metadata for Chroma
            metadata = {}

            # Add tags
            if 'tags' in ad_data:
                tags_data = ad_data['tags']
                metadata['tags'] = ','.join(tags_data.get('tags', []))
                metadata['type'] = tags_data.get('type', 'general')
                metadata['alignment'] = tags_data.get('alignment', '')
                metadata['theme'] = tags_data.get('theme', '')
                metadata['specific'] = tags_data.get('specific', '')

            # Add features
            if 'features' in ad_data:
                features = ad_data['features']
                metadata['sentiment'] = features.get('sentiment', 'neutral')
                metadata['has_cta'] = str(features.get('has_cta', False))
                metadata['dominant_color'] = features.get('dominant_color', 'unknown')
                metadata['brightness'] = str(features.get('brightness', 0.5))
                metadata['contrast'] = str(features.get('contrast', 0.5))

            # Add text content
            if 'text_content' in ad_data:
                metadata['description'] = ad_data['text_content'][:500]  # Limit length

            try:
                # Add to Chroma
                collection.add(
                    ids=[ad_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                loaded_count += 1

                if loaded_count % 10 == 0:
                    print(f"  Loaded {loaded_count} ads into Chroma...")

            except Exception as e:
                print(f"⚠ Failed to load ad {ad_id}: {e}")

    print(f"✓ Loaded {loaded_count} ads into Chroma collection '{collection_name}'")

def load_elasticsearch_database(
    metadata_file: str,
    es_url: str = "http://localhost:9200",
    index_name: str = "ads_metadata"
) -> None:
    """
    Load ad metadata into Elasticsearch for natural language search.

    Args:
        metadata_file: Path to JSONL file with processed ad data
        es_url: Elasticsearch URL
        index_name: Name of ES index
    """
    print("Loading data into Elasticsearch...")

    # Initialize Elasticsearch
    es_client = Elasticsearch(es_url)

    # Create index with mapping
    mapping = {
        "mappings": {
            "properties": {
                "ad_id": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "type": {"type": "keyword"},
                "alignment": {"type": "keyword"},
                "theme": {"type": "keyword"},
                "specific": {"type": "keyword"},
                "sentiment": {"type": "keyword"},
                "has_cta": {"type": "boolean"},
                "dominant_color": {"type": "keyword"},
                "brightness": {"type": "float"},
                "contrast": {"type": "float"},
                "description": {"type": "text"},
                "keywords": {"type": "keyword"},
                "text_length": {"type": "integer"},
                "word_count": {"type": "integer"}
            }
        }
    }

    try:
        es_client.indices.create(index=index_name, body=mapping, ignore=400)
        print(f"✓ Created Elasticsearch index '{index_name}'")
    except Exception as e:
        print(f"⚠ Index creation failed (may already exist): {e}")

    # Load data
    loaded_count = 0
    with open(metadata_file, 'r') as f:
        for line in f:
            ad_data = json.loads(line.strip())

            ad_id = ad_data['ad_id']

            # Prepare document for ES
            doc = {
                'ad_id': ad_id,
                'description': ad_data.get('text_content', ''),
            }

            # Add tags data
            if 'tags' in ad_data:
                tags_data = ad_data['tags']
                doc.update({
                    'tags': tags_data.get('tags', []),
                    'type': tags_data.get('type', 'general'),
                    'alignment': tags_data.get('alignment'),
                    'theme': tags_data.get('theme'),
                    'specific': tags_data.get('specific')
                })

            # Add features
            if 'features' in ad_data:
                features = ad_data['features']
                doc.update({
                    'sentiment': features.get('sentiment', 'neutral'),
                    'has_cta': features.get('has_cta', False),
                    'dominant_color': features.get('dominant_color', 'unknown'),
                    'brightness': features.get('brightness', 0.5),
                    'contrast': features.get('contrast', 0.5),
                    'text_length': features.get('text_length', 0),
                    'word_count': features.get('word_count', 0),
                    'keywords': features.get('keywords', [])
                })

            try:
                # Index document
                es_client.index(index=index_name, id=ad_id, document=doc)
                loaded_count += 1

                if loaded_count % 10 == 0:
                    print(f"  Indexed {loaded_count} ads in Elasticsearch...")

            except Exception as e:
                print(f"⚠ Failed to index ad {ad_id}: {e}")

    print(f"✓ Indexed {loaded_count} ads in Elasticsearch index '{index_name}'")

def main():
    """Main database loading execution."""
    print("=" * 60)
    print("PrivAds: Database Loading Pipeline")
    print("=" * 60)

    # Configuration
    metadata_file = "backend/data/ad_metadata.jsonl"
    chroma_path = "backend/chroma_db"
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    # Check if metadata file exists
    if not Path(metadata_file).exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Run ad processing pipeline first: python pipeline/run_ad_pipeline.py")
        return

    try:
        # Load into Chroma
        load_chroma_database(
            metadata_file=metadata_file,
            chroma_path=chroma_path
        )

        # Load into Elasticsearch
        load_elasticsearch_database(
            metadata_file=metadata_file,
            es_url=es_url
        )

        print("\n" + "=" * 60)
        print("✅ Database Loading Complete!")
        print("=" * 60)
        print(f"📊 Chroma DB: {chroma_path}")
        print(f"🔍 Elasticsearch: {es_url}")
        print("\nNext steps:")
        print("  - Start backend API: cd backend && python main.py")
        print("  - Test endpoints with sample requests")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Database loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()