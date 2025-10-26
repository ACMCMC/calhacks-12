"""
ML Model Training Pipeline
Trains logistic regression model with polynomial features on synthetic interaction data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from synthetic_generator import SyntheticInteractionGenerator
from sliding_window_extractor import SlidingWindowExtractor


class InteractionModelTrainer:
    """Trains ML model for predicting ad click probability from user interactions"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Initialize components
        self.generator = SyntheticInteractionGenerator(random_seed)
        self.extractor = SlidingWindowExtractor()

        # Model pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=random_seed, n_estimators=100, class_weight='balanced'))
        ])

        self.is_trained = False

    def generate_training_data(self, n_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate training data using synthetic generator and sliding window extraction"""
        print(f"Generating {n_samples} training samples...")

        all_features = []
        all_labels = []
        all_profiles = []

        samples_processed = 0

        while len(all_features) < n_samples:
            # Sample user profile
            profile = np.random.choice(list(self.generator.PROFILES.keys()))

            # Generate interaction sequence
            sequence = self.generator.generate_interaction_sequence(profile)

            # Extract sliding windows
            windows = self.extractor.extract_all_windows(sequence)

            for features, clicked_ad in windows:
                if len(all_features) >= n_samples:
                    break

                all_features.append(features.to_array())
                all_labels.append(int(clicked_ad))
                all_profiles.append(profile)

                samples_processed += 1
                if samples_processed % 10000 == 0:
                    print(f"Processed {samples_processed} windows, collected {len(all_features)} training samples...")

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"Generated {len(X)} training samples")
        print(f"Positive class ratio: {y.mean():.4f}")
        print(f"Profile distribution: {pd.Series(all_profiles).value_counts().to_dict()}")

        return X, y, all_profiles

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the ML model with class balancing"""
        print("Training model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )

        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Update model with class weights
        self.model.named_steps['classifier'].class_weight = class_weight_dict

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        test_pred_proba = self.model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)

        print(".4f")
        print(".4f")

        # Additional metrics
        test_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        print("\nConfusion Matrix:")
        print(cm)

        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation"""
        print(f"Performing {cv_folds}-fold cross-validation...")

        # Use AUC as scoring metric
        auc_scores = cross_val_score(
            self.model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )

        print(f"Cross-validation AUC scores: {auc_scores}")
        print(".4f")

        return {
            'cv_auc_scores': auc_scores,
            'cv_auc_mean': auc_scores.mean(),
            'cv_auc_std': auc_scores.std()
        }

    def save_model(self, model_path: str, scaler_path: str):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Save the full pipeline
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # Save scaler separately for potential client-side use
        scaler = self.model.named_steps['scaler']
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    def analyze_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze which features are most important for prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing features")

        classifier = self.model.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            # Tree-based model
            importances = classifier.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            top_features = [(feature_names[i], importances[i]) for i in sorted_indices[:20]]
        else:
            # Fallback for linear models
            coef = np.abs(classifier.coef_[0])
            sorted_indices = np.argsort(coef)[::-1]
            top_features = [(feature_names[i], coef[i]) for i in sorted_indices[:20]]

        print("\nTop 20 Most Important Features:")
        for name, importance in top_features:
            print(f"{name}: {importance:.4f}")

        return {
            'top_features': top_features,
            'all_importances': dict(zip(feature_names, importances if hasattr(classifier, 'feature_importances_') else coef))
        }

    def plot_feature_distributions(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Plot feature distributions for positive vs negative classes"""
        df = pd.DataFrame(X, columns=feature_names)
        df['clicked_ad'] = y

        # Only plot features that exist in the dataframe
        plot_features = [f for f in feature_names if f in df.columns]
        n_features = len(plot_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(8, 3 * n_features))
        if n_features == 1:
            axes = [axes]
        for i, feature in enumerate(plot_features):
            sns.histplot(data=df, x=feature, hue='clicked_ad', ax=axes[i], alpha=0.7)
            axes[i].set_title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.show()

    def run_full_pipeline(self, n_samples: int = 100000):
        """Run the complete training pipeline"""
        print("=== Starting Interaction Model Training Pipeline ===")

        # Generate training data
        X, y, profiles = self.generate_training_data(n_samples)

        # Cross-validation
        cv_results = self.cross_validate(X, y)

        # Train final model
        train_results = self.train_model(X, y)

        # Analyze features
        feature_importance = self.analyze_feature_importance(X, self.extractor.feature_names)

        # Plot distributions
        self.plot_feature_distributions(X, y, self.extractor.feature_names)

        # Save model
        model_path = '/home/acreomarino/privads/models/interaction_predictor.pkl'
        scaler_path = '/home/acreomarino/privads/models/interaction_scaler.pkl'

        Path('/home/acreomarino/privads/models').mkdir(exist_ok=True)
        self.save_model(model_path, scaler_path)

        # Save feature names
        self.extractor.save_feature_names('/home/acreomarino/privads/models/feature_names.json')

        # Compile results
        results = {
            'cv_results': cv_results,
            'train_results': train_results,
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'positive_ratio': y.mean(),
            'model_path': model_path,
            'scaler_path': scaler_path
        }

        # Save results summary
        with open('/home/acreomarino/privads/models/training_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif hasattr(obj, '__float__'):
                    return float(obj)
                else:
                    return str(obj)

            json.dump(results, f, default=convert_numpy, indent=2)

        print("=== Training Complete ===")
        print(f"Model saved to: {model_path}")

        return results


if __name__ == "__main__":
    # Run the full training pipeline
    trainer = InteractionModelTrainer()
    results = trainer.run_full_pipeline(n_samples=100000)  # Start with smaller dataset for testing

    print("\nTraining Summary:")
    print(f"Cross-validation AUC: {results['cv_results']['cv_auc_mean']:.4f} ± {results['cv_results']['cv_auc_std']:.4f}")
    print(f"Final test AUC: {results['train_results']['test_auc']:.4f}")
    print(f"Training samples: {results['n_samples']}")
    print(f"Positive class ratio: {results['positive_ratio']:.4f}")