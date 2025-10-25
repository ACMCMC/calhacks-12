# PrivAds

The Privacy-First AI Advertising Platform

## Overview

PrivAds learns user preferences **only from clicks**, without storing behavioral history or demographics. It uses:

1. **Jina CLIP v2** (frozen) to encode ads (text + optional images) into unified embeddings
2. **User embeddings** learned via contrastive loss on co-click graph
3. **Projector** to map ad space → user space
4. **Thompson Sampling** for exploration (coming in Phase 2)

## Architecture

```
Ad (text + image) → Jina CLIP v2 → z_ad → Projector → p_ad
                                                        ↓
User clicks → Co-click graph → InfoNCE → user_embeddings
                                                        ↓
                                    Scoring: cos(u, p_ad)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Phase 1 (End-to-end training)

```bash
python main.py
```

This will:
- Load Jina CLIP v2 (frozen)
- Generate synthetic click data
- Train user embeddings (InfoNCE on co-clicks)
- Train projector (ad space → user space)
- Save all models and embeddings

**Expected time:** ~5-10 min on CPU, ~2 min on GPU

### 3. Check outputs

```
models/
  ├── user_embeddings.npy      # Trained user vectors (1000, 128)
  ├── global_mean.npy           # Init for new users (128,)
  └── projector.pt              # Projector weights

data/
  ├── ad_embeddings_raw.npz     # z_ad from Jina CLIP (500, 1024)
  └── ad_projected.npz          # p_ad in user space (500, 128)
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
