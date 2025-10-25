"""
Project Aura - Automated Pipeline Runner
Runs all components in sequence, skipping completed steps.
"""

import sys
import os
from pathlib import Path

def check_models_exist():
    """Check if trained models exist."""
    models_dir = Path("backend/models")
    required_files = ["user_embeddings.npy", "global_mean.npy", "projector.pt"]
    return all((models_dir / f).exists() for f in required_files)

def check_ad_metadata_exists():
    """Check if ad metadata exists."""
    metadata_file = Path("backend/data/ad_metadata.jsonl")
    return metadata_file.exists()

def check_chroma_db_exists():
    """Check if Chroma database exists."""
    chroma_dir = Path("backend/chroma_db")
    return chroma_dir.exists() and len(list(chroma_dir.glob("*"))) > 0

def run_training_pipeline():
    """Run the PrivAds training pipeline."""
    print("🚀 Running PrivAds Training Pipeline...")
    exit_code = os.system("python pipeline/training/train_models.py")
    if exit_code != 0:
        print("❌ Training pipeline failed!")
        return False
    return True

def run_ad_processing():
    """Run the ad processing pipeline."""
    print("🎨 Running Ad Processing Pipeline...")
    exit_code = os.system("python pipeline/run_ad_pipeline.py")
    if exit_code != 0:
        print("❌ Ad processing pipeline failed!")
        return False
    return True

def run_database_loading():
    """Load data into databases."""
    print("💾 Loading Data into Databases...")
    exit_code = os.system("python pipeline/load_databases.py")
    if exit_code != 0:
        print("❌ Database loading failed!")
        return False
    return True

def start_backend():
    """Start the FastAPI backend."""
    print("🌐 To start the Backend API, run in a separate terminal:")
    print("   cd backend && python main.py")
    print("   API will be available at: http://localhost:8000")

def main():
    """Run the complete Project Aura pipeline."""
    print("\n" + "="*60)
    print("🎯 Project Aura: Automated Pipeline Runner")
    print("="*60)

    success = True

    # Step 1: Check and run training pipeline
    print("\n📊 Step 1: PrivAds Model Training")
    if check_models_exist():
        print("✅ Models already exist, skipping training")
    else:
        print("⚠️  Models not found, running training pipeline...")
        if not run_training_pipeline():
            success = False

    # Step 2: Check and run ad processing
    print("\n🎨 Step 2: Ad Processing")
    if check_ad_metadata_exists():
        print("✅ Ad metadata already exists, skipping processing")
    else:
        print("⚠️  Ad metadata not found, running ad processing...")
        if not run_ad_processing():
            success = False

    # Step 3: Check and run database loading
    print("\n💾 Step 3: Database Loading")
    if check_chroma_db_exists():
        print("✅ Chroma database already exists, skipping loading")
    else:
        print("⚠️  Chroma database not found, running database loading...")
        if not run_database_loading():
            success = False

    # Step 4: Instructions for starting backend API
    print("\n🌐 Step 4: Backend API")
    if success:
        print("✅ All components ready!")
        start_backend()
    else:
        print("❌ Some components failed. Please check the errors above.")
        print("You can try running individual components manually:")
        print("  - python pipeline/training/train_models.py")
        print("  - python pipeline/run_ad_pipeline.py")
        print("  - python pipeline/load_databases.py")
        print("  - cd backend && python main.py")
        sys.exit(1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
