"""
Main Pipeline Runner
Runs the complete interaction modeling pipeline from data generation to deployment.
"""

import sys
import os
from pathlib import Path

# Add the interaction_modeling directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from model_trainer import InteractionModelTrainer
from onnx_exporter import ONNXExporter


def run_full_pipeline(n_samples: int = 50000):
    """Run the complete pipeline: generate data → train model → export to ONNX"""

    print("🚀 Starting Complete Interaction Modeling Pipeline")
    print("=" * 60)

    # Step 1: Train the model
    print("\n📊 Step 1: Training ML Model")
    trainer = InteractionModelTrainer()
    results = trainer.run_full_pipeline(n_samples)

    # Step 2: Export to ONNX for browser deployment
    print("\n🌐 Step 2: Exporting to ONNX for Browser Deployment")
    exporter = ONNXExporter(
        model_path='/home/acreomarino/privads/models/interaction_predictor.pkl',
        scaler_path='/home/acreomarino/privads/models/interaction_scaler.pkl',
        feature_names_path='/home/acreomarino/privads/models/feature_names.json'
    )
    exporter.run_export_pipeline()

    # Step 3: Summary
    print("\n🎉 Pipeline Complete!")
    print("=" * 60)
    print("Results Summary:")
    print(f"  • Training samples: {results['n_samples']:,}")
    print(f"  • Positive class ratio: {results['positive_ratio']:.4f}")
    print(f"  • Cross-validation AUC: {results['cv_results']['cv_auc_mean']:.4f} ± {results['cv_results']['cv_auc_std']:.4f}")
    print(f"  • Final test AUC: {results['train_results']['test_auc']:.4f}")
    print("\nFiles created:")
    print("  • /models/interaction_predictor.pkl - Trained model")
    print("  • /models/interaction_predictor.onnx - Browser-compatible model")
    print("  • /models/interaction_scaler.pkl - Feature scaler")
    print("  • /models/feature_names.json - Feature metadata")
    print("  • /models/training_results.json - Training metrics")
    print("  • /privads-demo/public/models/ - Browser deployment files")

    return results


def run_quick_test():
    """Run a quick test with small dataset for development"""
    print("🧪 Running Quick Test Pipeline")
    return run_full_pipeline(n_samples=5000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Interaction Modeling Pipeline')
    parser.add_argument('--quick', action='store_true', help='Run quick test with small dataset')
    parser.add_argument('--samples', type=int, default=50000, help='Number of training samples')

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_full_pipeline(args.samples)