"""
ONNX Export for Browser Deployment
Converts trained model to ONNX format for client-side inference.
"""

import numpy as np
from pathlib import Path
import joblib
import json
from typing import List
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class ONNXExporter:
    """Exports trained ML model to ONNX format for browser deployment"""

    def __init__(self, model_path: str, scaler_path: str, feature_names_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path

    def load_model_components(self):
        """Load the trained model and supporting files"""
        print("Loading model components...")

        # Load model
        self.model = joblib.load(self.model_path)
        print(f"✓ Loaded model from {self.model_path}")

        # Load scaler
        self.scaler = joblib.load(self.scaler_path)
        print(f"✓ Loaded scaler from {self.scaler_path}")

        # Load feature names
        with open(self.feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"✓ Loaded {len(self.feature_names)} feature names")

    def create_simplified_model(self):
        """Create a simplified model without polynomial features for better browser performance"""
        print("Creating simplified model for browser deployment...")

        # Extract just the logistic regression (remove polynomial features)
        # This will make the model smaller and faster in the browser
        lr_model = self.model.named_steps['classifier']

        # Create new pipeline without polynomial features
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Copy the trained logistic regression
        simplified_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])

        # Copy fitted parameters
        simplified_model.named_steps['scaler'].mean_ = self.scaler.mean_
        simplified_model.named_steps['scaler'].scale_ = self.scaler.scale_
        simplified_model.named_steps['scaler'].var_ = self.scaler.var_

        simplified_model.named_steps['classifier'].coef_ = lr_model.coef_
        simplified_model.named_steps['classifier'].intercept_ = lr_model.intercept_
        simplified_model.named_steps['classifier'].classes_ = lr_model.classes_

        return simplified_model

    def export_to_onnx(self, output_path: str):
        """Export model to ONNX format"""
        print("Exporting to ONNX format...")

        # Use simplified model for browser
        model_to_export = self.create_simplified_model()

        # Define input shape (number of features)
        n_features = len(self.feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        # Convert to ONNX with older opset for browser compatibility
        onnx_model = convert_sklearn(
            model_to_export,
            initial_types=initial_type,
            target_opset=8,  # Use ONNX opset 8 for browser compatibility
            options={'zipmap': False}  # Better for browser performance
        )

        # Save ONNX model
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print(f"✓ ONNX model saved to {output_path}")
        print(f"  Model size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    def test_onnx_model(self, onnx_path: str, n_test_samples: int = 100):
        """Test the exported ONNX model"""
        print("Testing ONNX model...")

        # Create ONNX session
        session = ort.InferenceSession(onnx_path)

        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(n_test_samples, len(self.feature_names)).astype(np.float32)

        # Test inference
        correct_predictions = 0
        original_probs = self.model.predict_proba(test_data)

        for i in range(n_test_samples):
            # ONNX inference
            onnx_input = {session.get_inputs()[0].name: test_data[i:i+1]}
            onnx_output = session.run(None, onnx_input)[0]

            # Handle different output formats
            if hasattr(onnx_output, 'shape') and len(onnx_output.shape) > 1:
                onnx_prob = onnx_output[0][1]  # Probability of positive class
            else:
                onnx_prob = onnx_output[1] if len(onnx_output) > 1 else onnx_output[0]

            # Original model prediction
            original_prob = original_probs[i][1]

            # Check if predictions are close (within 1% tolerance)
            if abs(onnx_prob - original_prob) < 0.01:
                correct_predictions += 1

        accuracy = correct_predictions / n_test_samples
        print(".1f")

        if accuracy < 0.95:
            print("⚠ Warning: ONNX model predictions differ significantly from original!")
        else:
            print("✓ ONNX model predictions match original model")

    def create_browser_interface(self, onnx_path: str, output_dir: str):
        """Create JavaScript interface for browser deployment"""
        print("Creating browser interface...")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Copy ONNX model to web directory
        web_onnx_path = output_dir / 'interaction_predictor.onnx'
        import shutil
        shutil.copy2(onnx_path, web_onnx_path)

        # Create JavaScript interface
        js_code = f'''/**
 * Browser-based Interaction Prediction
 * Uses ONNX model for real-time ad click probability prediction
 */

class InteractionPredictor {{
    constructor() {{
        this.session = null;
        this.featureNames = {json.dumps(self.feature_names)};
        this.isReady = false;
    }}

    async initialize() {{
        try {{
            // Load ONNX model
            this.session = await ort.InferenceSession.create('/models/interaction_predictor.onnx');
            this.isReady = true;
            console.log('Interaction predictor initialized');
        }} catch (error) {{
            console.error('Failed to initialize predictor:', error);
            throw error;
        }}
    }}

    /**
     * Predict click probability from interaction features
     * @param {{Object}} features - Feature object with interaction metrics
     * @returns {{number}} Click probability (0-1)
     */
    async predictClickProbability(features) {{
        if (!this.isReady) {{
            throw new Error('Predictor not initialized');
        }}

        // Convert features to array in correct order
        const featureArray = this.featureNames.map(name => {{
            return features[name] !== undefined ? features[name] : 0;
        }});

        // Convert to tensor
        const tensor = new ort.Tensor('float32', featureArray, [1, featureArray.length]);

        // Run inference
        const results = await this.session.run({{ float_input: tensor }});

        // Return probability of positive class (click)
        return results.output.data[0];
    }}

    /**
     * Get feature names for debugging
     */
    getFeatureNames() {{
        return [...this.featureNames];
    }}
}}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = InteractionPredictor;
}} else if (typeof window !== 'undefined') {{
    window.InteractionPredictor = InteractionPredictor;
}}
'''

        js_path = output_dir / 'interaction_predictor.js'
        with open(js_path, 'w') as f:
            f.write(js_code)

        print(f"✓ JavaScript interface created at {js_path}")

        # Create HTML demo page
        html_code = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interaction Prediction Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"></script>
</head>
<body>
    <h1>Interaction Prediction Demo</h1>
    <div id="status">Loading...</div>
    <div id="prediction">Prediction: --</div>

    <script src="interaction_predictor.js"></script>
    <script>
        const predictor = new InteractionPredictor();

        async function init() {
            try {
                await predictor.initialize();
                document.getElementById('status').textContent = 'Model loaded successfully!';

                // Test with sample features
                const sampleFeatures = {
                    time_since_last_action: 2.5,
                    avg_time_between_actions: 3.2,
                    session_duration: 45.0,
                    scroll_down_count: 0.15,
                    scroll_up_count: 0.05,
                    click_count: 0.08,
                    hover_count: 0.12,
                    blur_count: 0.02,
                    focus_count: 0.03,
                    wait_count: 0.10,
                    close_tab_count: 0.001,
                    scroll_depth_max: 0.8,
                    interaction_density: 0.25,
                    attention_score: 0.75,
                    scroll_velocity_avg: 0.15,
                    click_to_hover_ratio: 0.67,
                    blur_frequency: 0.02,
                    action_entropy: 1.8,
                    burstiness_score: 0.9,
                    engagement_rhythm: 0.3
                };

                const probability = await predictor.predictClickProbability(sampleFeatures);
                document.getElementById('prediction').textContent =
                    `Prediction: ${(probability * 100).toFixed(2)}% click probability`;

            } catch (error) {
                document.getElementById('status').textContent = `Error: ${error.message}`;
                console.error(error);
            }
        }

        init();
    </script>
</body>
</html>'''

        html_path = output_dir / 'demo.html'
        with open(html_path, 'w') as f:
            f.write(html_code)

        print(f"✓ Demo HTML page created at {html_path}")

    def run_export_pipeline(self):
        """Run the complete export pipeline"""
        print("=== Starting ONNX Export Pipeline ===")

        # Load components
        self.load_model_components()

        # Export to ONNX
        onnx_path = '/home/acreomarino/privads/models/interaction_predictor.onnx'
        self.export_to_onnx(onnx_path)

        # Test ONNX model
        self.test_onnx_model(onnx_path)

        # Create browser interface
        web_dir = '/home/acreomarino/privads/privads-demo/public/models'
        Path(web_dir).mkdir(parents=True, exist_ok=True)
        self.create_browser_interface(onnx_path, web_dir)

        print("\n=== Export Complete ===")
        print(f"ONNX model: {onnx_path}")
        print(f"Browser files: {web_dir}/")
        print("Ready for client-side deployment!")


if __name__ == "__main__":
    exporter = ONNXExporter(
        model_path='/home/acreomarino/privads/models/interaction_predictor.pkl',
        scaler_path='/home/acreomarino/privads/models/interaction_scaler.pkl',
        feature_names_path='/home/acreomarino/privads/models/feature_names.json'
    )

    exporter.run_export_pipeline()