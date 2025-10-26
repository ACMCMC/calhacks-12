# PrivAds: AI-Native Growth Platform

**Reimagining Mixpanel for the AI era** - An active, privacy-first growth platform that functions as both an intelligent ad server and analytics engine.

## Overview

PrivAds uses a **3-part AI system** to understand user preferences privately (`PrivAds`), predict real-time user receptiveness (on-device), and understand page context (on-device) to serve hyper-personalized ads. The backend provides APIs for ad serving, search, and analytics while maintaining strict privacy principles.

### Core Features
- **Privacy-First Preference Matching**: Learns user preferences from co-click data only
- **Real-Time Ad Serving API**: Combines user preference, receptiveness, and context
- **Natural Language Ad Search**: Search ads by metadata and features
- **Vector Database Integration**: Chroma for embedding storage and retrieval
- **Elasticsearch Integration**: Natural language search over ad metadata

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ad Creative   │ -> │   Jina CLIP v2   │ -> │   Projector     │
│  (Text+Image)   │    │    (Frozen)      │    │   (Trained)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                           ↓                   ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Clicks    │ -> │  Co-click Graph  │ -> │ User Embeddings │
│   (Synthetic)   │    │   (InfoNCE)      │    │   (128D)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ↓                           ↓                   ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chroma DB     │    │ Elasticsearch    │    │   FastAPI       │
│ (Embeddings)    │    │  (Metadata)      │    │   Backend       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
# Train the PrivAds models (user embeddings + projector)
python pipeline/training/train_models.py

# Process ad creatives and load databases
python pipeline/run_ad_pipeline.py
python pipeline/load_databases.py
```

### 3. Start Backend API

```bash
cd backend
python main.py
```

### 4. Test API Endpoints

```bash
# Get personalized ad
curl -X POST http://localhost:8000/get_ad \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "p_receptive": 0.8,
    "site_context": {
      "type": "political",
      "alignment": "conservative",
      "keywords": ["republican", "gop"]
    }
  }'

# Search ads by natural language
curl -X POST http://localhost:8000/search_ads \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me car ads that are videos"}'
```

## Project Structure

```
privads/
├── backend/                    # FastAPI backend
│   ├── main.py                # API endpoints (/get_ad, /search_ads)
│   ├── privads_core.py        # PrivAds logic + Chroma interface
│   ├── elastic_search.py      # Elasticsearch/Agent Builder interface
│   └── models/                # Trained models (.pt, .npy files)
├── pipeline/                  # Data processing pipeline
│   ├── scrape_ads.py         # Bright Data integration
│   ├── run_ad_pipeline.py    # Main ad processing script
│   ├── load_databases.py     # Load data into Chroma/ES
│   └── training/             # Model training
│       ├── generate_clicks.py # Synthetic click generation
│       ├── train_user_embeddings.py
│       ├── train_projector.py
│       └── train_models.py   # Orchestration script
├── ad_processing/            # Feature extraction
│   ├── encoder.py           # Jina CLIP wrapper
│   ├── feature_extractor.py # Metadata extraction
│   └── tagger.py           # Contextual tagging
├── requirements.txt          # Python dependencies
└── README.md
```

## API Endpoints

### POST `/get_ad`
Serves personalized ads based on user preference, receptiveness, and context.

**Request:**
```json
{
  "user_id": "string",
  "p_receptive": 0.8,
  "site_context": {
    "type": "political",
    "alignment": "conservative",
    "keywords": ["republican", "gop"]
  }
}
```

**Response:**
```json
{
  "decision": "SERVE",
  "ad_id": "ad001",
  "creative_url": "https://...",
  "score": 0.85
}
```

### POST `/search_ads`
Natural language search over ad metadata.

**Request:**
```json
{
  "query": "Show me car ads that are videos"
}
```

**Response:**
```json
{
  "results": [
    {
      "ad_id": "ad001",
      "thumbnail_url": "https://...",
      "metadata": {...}
    }
  ]
}
```

## Technology Stack

- **Backend**: FastAPI, Python
- **ML**: PyTorch, Jina CLIP v2, scikit-learn
- **Databases**: Chroma (vectors), Elasticsearch (search)
- **Scraping**: Bright Data API
- **Deployment**: Fly.io / Render / AWS EC2

## Development Setup

### Prerequisites
- Python 3.8+
- PyTorch with CUDA (recommended)
- Elasticsearch instance
- Chroma database

### Environment Setup
```bash
# Clone and setup
git clone <repo>
cd privads
pip install -r requirements.txt

# Set environment variables
export ELASTICSEARCH_URL="http://localhost:9200"
export CHROMA_URL="http://localhost:8001"
export BRIGHT_DATA_API_KEY="your_key"
```

### Running Tests
```bash
# Test components
python -m pytest tests/

# Test API
python -m pytest tests/test_api.py
```

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints and docstrings

## License

MIT License - see LICENSE file for details.
  ├── user_embeddings.npy      # Trained user vectors (1000, 128)
  ├── global_mean.npy           # Init for new users (128,)
  └── projector.pt              # Projector weights

data/
  ├── ad_embeddings_raw.npz     # z_ad from Jina CLIP (500, 768)
  └── ad_projected.npz          # p_ad in user space (500, 128)

evaluation_results/
  ├── precision_10.txt          # Precision@10 score
  ├── user_embeddings_tsne.png  # User embedding visualization
  └── recommendation_heatmap.png # User-ad similarity heatmap
```

## Project Structure

```
privads/
├── main.py                      # Full pipeline
├── requirements.txt
├── src/
│   ├── ad_encoder.py           # Jina CLIP v2 wrapper
│   ├── projector.py            # MLP: ad space → user space
│   ├── click_data.py           # Synthetic/real click sources
│   ├── train_user_embeddings.py # InfoNCE on co-clicks
│   └── train_projector.py      # InfoNCE + centroid loss
├── data/                       # Embeddings and datasets
├── models/                     # Trained models
└── notebooks/                  # Experiments
```

## Testing Individual Components

```bash
# Test ad encoder
cd src && python ad_encoder.py

# Test projector
python projector.py

# Test synthetic clicks
python click_data.py

# Test user embedding training
python train_user_embeddings.py

# Test projector training
python train_projector.py
```

## Swapping to Real Data

Edit `main.py` and replace:

```python
# Before
click_gen = SyntheticClickGenerator(...)

# After
from click_data import RealClickData
click_gen = RealClickData(data_path="prod_data/clicks.csv")
```

Your CSV should have columns: `user_id`, `ad_id`, `clicked`, `position`

## Privacy Guarantees

- ✅ No user metadata stored (no age, gender, location, etc.)
- ✅ Only embeddings: `user_id → vector (128,)`
- ✅ Embeddings are abstract—can't reverse to click history
- ✅ Can delete user on request (GDPR compliant)
- ✅ Optional: Add DP-SGD noise to updates

## Next Steps (Phase 2+)

- [ ] Thompson Sampling (LinTS with position bias)
- [ ] Serving API (rank ads for user)
- [ ] Feedback loop (update embeddings online)
- [ ] Real ad data with images
- [ ] Evaluation metrics (Recall@K, NDCG)
- [ ] Cold-start strategies
- [ ] ANN index for fast retrieval (FAISS)

## License

Apache 2.0
