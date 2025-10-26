# Interaction Modeling System

Advanced ML-based system for predicting ad click probability from real-time user interactions using sliding window analysis and synthetic data generation.

## 🏗️ Architecture

```
Synthetic Data Generator → Sliding Window Extractor → ML Model → ONNX Export → Browser Deployment
```

## 📁 Components

### `synthetic_generator.py`
Generates realistic user interaction sequences using probabilistic models:
- **6 User Profiles**: rushed_reader, careful_reader, distracted_user, focused_reader, click_happy, mobile_user
- **9 Action Types**: scroll_down, scroll_up, click, hover, wait, focus_window, blur_window, close_tab, **click_ad**
- **Beta Distributions**: Model different behavioral patterns probabilistically
- **State-Dependent Actions**: Action probabilities change based on time since last interaction

### `sliding_window_extractor.py`
Converts interaction sequences into ML-ready feature vectors:
- **2-Minute Windows**: Captures complete user engagement cycles
- **20 Features**: Temporal, behavioral, and derived interaction metrics
- **Real-time Processing**: Optimized for live feature extraction

### `model_trainer.py`
Trains logistic regression model with polynomial features:
- **Class Balancing**: Handles rare ad-click events
- **Cross-Validation**: Robust performance evaluation
- **Feature Analysis**: Identifies most predictive interaction patterns

### `onnx_exporter.py`
Exports trained model for browser deployment:
- **ONNX Format**: Cross-platform ML model format
- **Simplified Architecture**: Optimized for browser performance
- **JavaScript Interface**: Ready-to-use browser API

## 🚀 Usage

### Quick Test
```bash
cd backend/interaction_modeling
python run_pipeline.py --quick  # 5,000 samples
```

### Full Training
```bash
python run_pipeline.py --samples 50000  # 50,000 samples
```

### Manual Testing
```bash
# Test synthetic generator
python synthetic_generator.py

# Test feature extractor
python sliding_window_extractor.py

# Test model training
python -c "from model_trainer import InteractionModelTrainer; trainer = InteractionModelTrainer(); trainer.run_full_pipeline(1000)"
```

## 📊 Features

### Temporal Features
- `time_since_last_action`: Seconds since last interaction (key attention signal)
- `avg_time_between_actions`: Mean interaction intervals
- `session_duration`: Total time on page

### Behavioral Features
- `scroll_down_count`, `scroll_up_count`: Scrolling patterns
- `click_count`, `hover_count`: Engagement actions
- `blur_count`, `focus_count`: Window attention changes
- `scroll_depth_max`: How far user scrolled

### Derived Features
- `interaction_density`: Actions per second
- `attention_score`: Focus vs distraction balance
- `action_entropy`: Diversity of user actions
- `burstiness_score`: Clustering of interactions
- `engagement_rhythm`: Interaction regularity

## 🎯 Model Performance

- **AUC Target**: > 0.75 on cross-validation
- **Class Imbalance**: ~6% positive class (ad clicks)
- **Browser Compatible**: < 50KB ONNX model size

## 🌐 Browser Deployment

After training, the system creates:
- `interaction_predictor.onnx`: Browser-compatible model
- `interaction_predictor.js`: JavaScript interface
- `demo.html`: Test page

### Usage in Browser
```javascript
const predictor = new InteractionPredictor();
await predictor.initialize();

const features = {
    time_since_last_action: 2.5,
    interaction_density: 0.25,
    attention_score: 0.75,
    // ... other features
};

const clickProbability = await predictor.predictClickProbability(features);
```

## 🔧 Configuration

### Window Settings
```python
WINDOW_SIZE = 120  # 2 minutes
WINDOW_STEP = 10   # 10 second steps
MIN_ACTIONS = 5    # Minimum actions per window
```

### User Profiles
Each profile defined by Beta distribution parameters for:
- Session length
- Interaction frequency
- Pause probability
- Click probability
- Ad click multiplier

## 📈 Results

After running the pipeline:
- **Models saved**: `/models/interaction_predictor.pkl`
- **Browser files**: `/privads-demo/public/models/`
- **Metrics**: `/models/training_results.json`

## 🎨 Integration

Replace the current heuristic-based prediction in `PrivAdsProvider.tsx`:

```typescript
// Old: Heuristic prediction
const prediction = calculateHeuristicPrediction(interactions);

// New: ML-based prediction
const features = extractSlidingWindowFeatures(interactions);
const prediction = await mlPredictor.predictClickProbability(features);
```

This provides **privacy-preserving**, **real-time**, **ML-powered** ad click prediction directly in the browser! 🚀