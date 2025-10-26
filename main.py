"""
PrivAds - Automated Pipeline Runner
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

def check_interaction_model_exists():
    """Check if interaction ML model exists."""
    models_dir = Path("backend/models")
    required_files = ["interaction_predictor.pkl", "interaction_scaler.pkl", "feature_names.json"]
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
    exit_code = os.system("cd /home/acreomarino/privads && python pipeline/training/train_models.py")
    if exit_code != 0:
        print("❌ Training pipeline failed!")
        return False
    return True

def run_ad_processing():
    """Run the ad processing pipeline."""
    print("🎨 Running Ad Processing Pipeline...")
    exit_code = os.system("cd /home/acreomarino/privads && python pipeline/run_ad_pipeline.py")
    if exit_code != 0:
        print("❌ Ad processing pipeline failed!")
        return False
    return True

def run_interaction_modeling():
    """Run the interaction modeling training pipeline."""
    print("🤖 Running Interaction Modeling Pipeline...")
    exit_code = os.system("cd backend/interaction_modeling && python run_pipeline.py --samples 10000")
    if exit_code != 0:
        print("❌ Interaction modeling pipeline failed!")
        return False
    return True

def run_database_loading():
    """Load data into databases."""
    print("💾 Loading Data into Databases...")
    exit_code = os.system("cd /home/acreomarino/privads && python pipeline/load_databases.py")
    if exit_code != 0:
        print("❌ Database loading failed!")
        return False
    return True

def start_backend():
    """Start the FastAPI backend."""
    print("🌐 To start the Backend API, run in a separate terminal:")
    print("   cd backend && python main.py")
    print("   API will be available at: http://localhost:8000")

def deploy_to_baseten(truss_dir, publish=False):
    """
    Deploy a Truss model to Baseten. If publish=True, deploys to production.
    Requires truss CLI and Baseten API key configured.
    """
    import subprocess
    cmd = ["truss", "push"]
    if publish:
        cmd.append("--publish")
    cmd.append(truss_dir)
    print(f"Deploying model to Baseten with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Baseten deployment failed:")
        print(result.stderr)
        return False
    print("Model deployed to Baseten successfully.")
    return True

def main():
    """Run the complete PrivAds pipeline."""
    print("\n" + "="*60)
    print("🎯 PrivAds: Automated Pipeline Runner")
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

    # Step 2: Check and run interaction modeling
    print("\n🤖 Step 2: Interaction Modeling")
    if check_interaction_model_exists():
        print("✅ Interaction model already exists, skipping training")
    else:
        print("⚠️  Interaction model not found, running interaction modeling...")
        if not run_interaction_modeling():
            success = False

    # Step 3: Check and run ad processing
    print("\n🎨 Step 3: Ad Processing")
    if check_ad_metadata_exists():
        print("✅ Ad metadata already exists, skipping processing")
    else:
        print("⚠️  Ad metadata not found, running ad processing...")
        if not run_ad_processing():
            success = False

    # Step 4: Check and run database loading
    print("\n💾 Step 4: Database Loading")
    if check_chroma_db_exists():
        print("✅ Chroma database already exists, skipping loading")
    else:
        print("⚠️  Chroma database not found, running database loading...")
        if not run_database_loading():
            success = False

    # Step 5: Instructions for starting backend API
    print("\n🌐 Step 5: Backend API")
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

    # Deploy trained model to Baseten
    truss_dir = "path/to/your/truss_model"  # Update with your actual Truss directory
    deploy_to_baseten(truss_dir, publish=True)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
