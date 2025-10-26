/**
 * Browser-based Interaction Prediction
 * Uses ONNX model for real-time ad click probability prediction
 */

class InteractionPredictor {
    constructor() {
        this.session = null;
        this.featureNames = ["time_since_last_action", "avg_time_between_actions", "session_duration", "scroll_down_count", "scroll_up_count", "scroll_depth_max", "interaction_density", "attention_score", "scroll_velocity_avg", "action_entropy", "burstiness_score", "engagement_rhythm"];
        this.isReady = false;
    }

    async initialize() {
        try {
            // Load ONNX model
            this.session = await ort.InferenceSession.create('/models/interaction_predictor.onnx');
            this.isReady = true;
            console.log('Interaction predictor initialized');
        } catch (error) {
            console.error('Failed to initialize predictor:', error);
            throw error;
        }
    }

    /**
     * Predict click probability from interaction features
     * @param {Object} features - Feature object with interaction metrics
     * @returns {number} Click probability (0-1)
     */
    async predictClickProbability(features) {
        if (!this.isReady) {
            throw new Error('Predictor not initialized');
        }

        // Convert features to array in correct order
        const featureArray = this.featureNames.map(name => {
            return features[name] !== undefined ? features[name] : 0;
        });

        // Convert to tensor
        const tensor = new ort.Tensor('float32', featureArray, [1, featureArray.length]);

        // Run inference
        const results = await this.session.run({ float_input: tensor });

        // Return probability of positive class (click)
        return results.output.data[0];
    }

    /**
     * Get feature names for debugging
     */
    getFeatureNames() {
        return [...this.featureNames];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InteractionPredictor;
} else if (typeof window !== 'undefined') {
    window.InteractionPredictor = InteractionPredictor;
}
