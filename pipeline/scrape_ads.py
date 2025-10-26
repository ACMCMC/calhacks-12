"""
Ad Scraping Pipeline for Project Aura
Uses Bright Data to scrape ad creatives from target websites.
"""

"""
YOU MUST DOWNLOAD THE HUGGING FACE FIRST HALF OF THE DATASET ON YOUR OWN COMPUTER
https://huggingface.co/datasets/PeterBrendan/AdImageNet/tree/main/data
ONLY DOWNLOAD THE FIRST ONE, not both.
"""

import os
import requests
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from dotenv import load_dotenv
import pandas as pd
import random
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

# Configuration for ad image extraction
MAX_AD_IMAGES = 100  # Maximum number of images to extract from HuggingFace dataset

import os
import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import base64

class BrightDataScraper:
    """Bright Data Web Scraper API integration for scraping ad creatives."""

    def __init__(self, api_key: str = None, dataset_id: str = None):
        self.api_key = api_key or os.getenv("BRIGHT_DATA_API_KEY")
        self.dataset_id = dataset_id or os.getenv("BRIGHT_DATA_DATASET_ID", "gd_l7q7dkf244hwjntr0")  # Default general scraper
        self.base_url = "https://api.brightdata.com"

        if not self.api_key:
            print("⚠ No Bright Data API key found. Set BRIGHT_DATA_API_KEY environment variable.")
            self.api_key = None

    def scrape_ads(
        self,
        target_urls: List[str],
        output_dir: str = "ad_creatives",
        max_ads_per_url: int = 10,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Scrape ad creatives from target websites using Bright Data API.

        Args:
            target_urls: List of website URLs to scrape
            output_dir: Directory to save downloaded ads
            max_ads_per_url: Maximum ads to collect per URL
            format_type: Output format (json, ndjson)

        Returns:
            Metadata about scraped ads
        """
        if not self.api_key:
            print("Bright Data API key required for scraping")
            return self._create_mock_data(target_urls, output_dir)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        scraped_data = {
            'ads': [],
            'metadata': {
                'total_scraped': 0,
                'sources': target_urls,
                'bright_data_job_id': None
            }
        }

        print(f"🚀 Starting Bright Data scraping from {len(target_urls)} sources...")

        # Prepare input data for Bright Data API
        input_data = []
        for url in target_urls:
            input_data.append({
                "url": url
                # Removed limit_per_input as it's not accepted by the API
            })

        # Trigger scraping job
        job_response = self._trigger_scraping_job(input_data, format_type)
        if not job_response:
            print("❌ Failed to trigger scraping job")
            return self._create_mock_data(target_urls, output_dir)

        job_id = job_response.get('collection_id') or job_response.get('snapshot_id')
        scraped_data['metadata']['bright_data_job_id'] = job_id

        print(f"✅ Job triggered successfully. Job ID: {job_id}")

        # Wait for job completion and get results
        results = self._wait_for_results(job_id, format_type)
        if results:
            scraped_data = self._process_results(results, output_path)
        else:
            print("⚠️ Unable to retrieve results directly from API")
            print("💡 Bright Data may deliver results via webhook or email")
            print(f"🔗 Job ID: {job_id} - Check your Bright Data dashboard for results")
            print("🔄 Falling back to mock data for development")
            scraped_data = self._create_mock_data(target_urls, output_dir)

        # Save metadata
        metadata_file = output_path / "scraped_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(scraped_data, f, indent=2)

        print(f"✓ Saved metadata to {metadata_file}")
        return scraped_data

    def _trigger_scraping_job(self, input_data: List[Dict], format_type: str) -> Dict[str, Any]:
        """Trigger a Bright Data scraping job."""
        url = f"{self.base_url}/datasets/v3/trigger"

        params = {
            'dataset_id': self.dataset_id,
            'format': format_type,
            'uncompressed_webhook': 'true'
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=input_data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error triggering scraping job: {e}")
            return None

    def _wait_for_results(self, job_id: str, format_type: str, max_wait_time: int = 60) -> List[Dict]:
        """Wait for scraping job to complete and get results."""
        print(f"⏳ Waiting for job {job_id} to complete (max {max_wait_time}s)...")

        start_time = time.time()
        check_count = 0

        while time.time() - start_time < max_wait_time:
            check_count += 1
            print(f"🔍 Check #{check_count} for job {job_id}...")

            results = self._get_job_results(job_id, format_type)
            if results:
                print("✅ Scraping job completed!")
                return results

            # Wait 15 seconds between checks
            time.sleep(15)

        print(f"⏰ Timeout waiting for scraping job to complete after {max_wait_time}s")
        return None

    def _get_job_results(self, job_id: str, format_type: str) -> List[Dict]:
        """Get results from a completed scraping job."""
        # Try different possible endpoints for Bright Data
        possible_endpoints = [
            f"{self.base_url}/datasets/v3/dataset/{job_id}",
            f"{self.base_url}/datasets/v3/trigger/{job_id}",
            f"{self.base_url}/scraping/v1/results/{job_id}"
        ]

        for url in possible_endpoints:
            try:
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }

                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if format_type == 'json':
                        return data
                    elif format_type == 'ndjson':
                        # Parse NDJSON
                        lines = response.text.strip().split('\n')
                        return [json.loads(line) for line in lines if line.strip()]
                else:
                    print(f"⚠️ Endpoint {url} returned {response.status_code}")

            except Exception as e:
                print(f"⚠️ Error with endpoint {url}: {e}")
                continue

        print(f"❌ Failed to get results from any endpoint")
        return None

    def _process_results(self, results: List[Dict], output_path: Path) -> Dict[str, Any]:
        """Process scraping results and save to files."""
        processed_data = {
            'ads': [],
            'metadata': {
                'total_scraped': 0,
                'sources': [],
                'bright_data_job_id': None
            }
        }

        for i, result in enumerate(results):
            # Extract ad data from result
            ad_data = self._extract_ad_data(result, i, output_path)
            if ad_data:
                processed_data['ads'].append(ad_data)
                processed_data['metadata']['total_scraped'] += 1

                # Track sources
                source = result.get('input', {}).get('url', 'unknown')
                if source not in processed_data['metadata']['sources']:
                    processed_data['metadata']['sources'].append(source)

        print(f"✓ Processed {processed_data['metadata']['total_scraped']} ads")
        return processed_data

    def _extract_ad_data(self, result: Dict, index: int, output_path: Path) -> Dict[str, Any]:
        """Extract ad data from a scraping result."""
        try:
            # This depends on the scraper configuration
            # Adjust based on what your Bright Data scraper returns
            ad_data = {
                'id': f"scraped_ad_{index:04d}",
                'source_url': result.get('input', {}).get('url', ''),
                'title': result.get('title', ''),
                'description': result.get('description', ''),
                'image_url': result.get('image_url', ''),
                'local_path': None
            }

            # Download image if available
            if ad_data['image_url']:
                image_path = self._download_image(ad_data['image_url'], ad_data['id'], output_path)
                ad_data['local_path'] = str(image_path) if image_path else None

            return ad_data

        except Exception as e:
            print(f"⚠️ Error extracting ad data: {e}")
            return None

    def _download_image(self, image_url: str, ad_id: str, output_path: Path) -> Path:
        """Download an image from URL."""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                # Determine file extension
                content_type = response.headers.get('content-type', '')
                ext = '.jpg'
                if 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'

                image_path = output_path / f"{ad_id}{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)

                return image_path

        except Exception as e:
            print(f"⚠️ Failed to download image {image_url}: {e}")

        return None

    def _extract_images_from_parquet(self, parquet_path: str, output_path: Path, max_images: int = MAX_AD_IMAGES) -> List[Dict[str, Any]]:
        """
        Extract images from HuggingFace adImageNet parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            output_path: Directory to save extracted images
            max_images: Maximum number of images to extract
            
        Returns:
            List of ad data dictionaries
        """
        try:
            print(f"📊 Loading HuggingFace adImageNet dataset from {parquet_path}")
            df = pd.read_parquet(parquet_path)
            
            print(f"📈 Dataset loaded: {len(df)} total images available")
            
            # Sample random images up to max_images
            sample_size = min(max_images, len(df))
            sampled_df = df.sample(n=sample_size, random_state=42)
            
            print(f"🎯 Extracting {sample_size} random images...")
            
            extracted_ads = []
            
            for idx, row in sampled_df.iterrows():
                try:
                    # Generate unique ad ID
                    ad_id = f"hf_ad_{idx:06d}"
                    
                    # Extract image data (HuggingFace datasets store images as dict with 'bytes' key)
                    if 'image' in row and row['image'] is not None:
                        image_data = row['image']
                        
                        # Handle HuggingFace dataset format: {'bytes': b'...', 'path': '...'}
                        if isinstance(image_data, dict) and 'bytes' in image_data:
                            image_bytes = image_data['bytes']
                            image_path = output_path / f"{ad_id}.jpg"
                            with open(image_path, 'wb') as f:
                                f.write(image_bytes)
                        # If it's a PIL Image, save it directly
                        elif hasattr(image_data, 'save'):
                            image_path = output_path / f"{ad_id}.jpg"
                            image_data.save(image_path, 'JPEG')
                        # If it's bytes data, write it directly
                        elif isinstance(image_data, bytes):
                            image_path = output_path / f"{ad_id}.jpg"
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                        else:
                            print(f"⚠️ Unknown image format for ad {ad_id}: {type(image_data)}, skipping")
                            continue
                        
                        # Extract description/caption if available
                        description = ""
                        if 'text' in row and pd.notna(row['text']):
                            description = str(row['text'])
                        elif 'caption' in row and pd.notna(row['caption']):
                            description = str(row['caption'])
                        elif 'description' in row and pd.notna(row['description']):
                            description = str(row['description'])
                        else:
                            description = f"Ad image from HuggingFace dataset (ID: {idx})"
                        
                        # Create ad data
                        ad_data = {
                            'id': ad_id,
                            'description': description,
                            'source_url': f"huggingface_dataset_row_{idx}",
                            'local_path': str(image_path),
                            'dataset_source': 'huggingface_adimagenet'
                        }
                        
                        extracted_ads.append(ad_data)
                        
                    else:
                        print(f"⚠️ No image data found for row {idx}, skipping")
                        
                except Exception as e:
                    print(f"⚠️ Error processing row {idx}: {e}")
                    continue
            
            print(f"✅ Successfully extracted {len(extracted_ads)} images from HuggingFace dataset")
            return extracted_ads
            
        except FileNotFoundError:
            print(f"❌ Parquet file not found: {parquet_path}")
            return []
        except Exception as e:
            print(f"❌ Error reading parquet file: {e}")
            return []

    def _create_mock_data(self, target_urls: List[str], output_dir: str) -> Dict[str, Any]:
        """Create mock data using HuggingFace datasets library (AdImageNet)."""
        print("🔧 Setting up the Hugging Face ads (100) using datasets library")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Download AdImageNet from HuggingFace
        dataset = load_dataset("PeterBrendan/AdImageNet", split="train[:100]")
        print(f"Loaded {len(dataset)} ads from HuggingFace AdImageNet")

        extracted_ads = []
        for i, row in enumerate(dataset):
            ad_id = f"hf_ad_{i:06d}"
            image = row['image']
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image_path = output_path / f"{ad_id}.jpg"
            image.save(image_path)
            description = row.get('caption') or row.get('text') or row.get('description') or f"Ad image {i}"
            ad_data = {
                'id': ad_id,
                'description': description,
                'source_url': f"huggingface_dataset_row_{i}",
                'local_path': str(image_path),
                'dataset_source': 'huggingface_adimagenet'
            }
            extracted_ads.append(ad_data)

        metadata = {
            'ads': extracted_ads,
            'metadata': {
                'total_scraped': len(extracted_ads),
                'sources': target_urls,
                'mock_data': False,
                'data_source': 'huggingface_adimagenet'
            }
        }
        metadata_file = output_path / "scraped_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        return metadata

    def _create_placeholder_image(self, path: Path, text: str):
        """Create a simple placeholder image."""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create a simple colored rectangle with text
            img = Image.new('RGB', (400, 300), color='lightblue')
            draw = ImageDraw.Draw(img)

            # Try to use default font, fallback to basic
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Draw text
            draw.text((200, 150), text, fill='black', anchor='mm', font=font)
            img.save(path)

        except ImportError:
            print("⚠ PIL not available, skipping placeholder image creation")
        except Exception as e:
            print(f"⚠ Failed to create placeholder image: {e}")

def main():
    """
    Main scraping execution.
    
    This script will:
    1. Try to scrape ads using Bright Data API if configured
    2. If Bright Data is not available, extract images from HuggingFace adImageNet parquet file
    3. If parquet file is not found, create placeholder mock images
    
    The number of images extracted from the parquet file is controlled by MAX_AD_IMAGES variable.
    """
    print("=" * 60)
    print("Project Aura: Ad Scraping Pipeline")
    print("=" * 60)

    # Configuration
    target_urls = [
        "https://www.amazon.com/s?k=sustainable+clothing",
        "https://www.amazon.com/s?k=luxury+watches", 
        "https://www.amazon.com/s?k=kids+toys",
        "https://www.amazon.com/s?k=organic+food",
        "https://www.amazon.com/s?k=gaming+laptops"
    ]

    output_dir = "ad_creatives"

    # Initialize scraper with optional custom dataset ID
    dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID")  # Custom scraper if available
    scraper = BrightDataScraper(dataset_id=dataset_id)

    try:
        results = scraper.scrape_ads(
            target_urls=target_urls,
            output_dir=output_dir,
            max_ads_per_url=5,
            format_type="json"
        )

        print("\n" + "=" * 60)
        print("✅ Ad Scraping Complete!")
        print("=" * 60)
        print(f"📁 Ads saved to: {output_dir}")
        print(f"📊 Total ads scraped: {results['metadata']['total_scraped']}")
        if results['metadata'].get('bright_data_job_id'):
            print(f"🔗 Bright Data Job ID: {results['metadata']['bright_data_job_id']}")
        print("\nNext steps:")
        print("  - Process ads: python pipeline/run_ad_pipeline.py")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
