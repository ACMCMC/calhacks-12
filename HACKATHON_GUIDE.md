# PrivAds Hackathon Quick Start

## What We Built (Phase 1)

✅ **Complete end-to-end pipeline** using:
- **Jina CLIP v2** (frozen) for multimodal ad embeddings (text + image → single vector)
- **User embeddings** learned from clicks only (no demographics!)
- **Projector** to align ad space with user space
- **Modular synthetic data** (easy swap to real data)

## Files Created

```
privads/
├── main.py                          # 🚀 Run this for full pipeline
├── test_components.py               # 🧪 Quick component tests
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
├── HACKATHON_GUIDE.md              # This file
│
├── src/
│   ├── ad_encoder.py               # Jina CLIP v2 wrapper
│   ├── projector.py                # MLP: z_ad → p_ad
│   ├── click_data.py               # Synthetic + real data interface
│   ├── train_user_embeddings.py   # InfoNCE on co-clicks
│   └── train_projector.py         # InfoNCE + centroid loss
│
├── data/                           # Will contain embeddings
├── models/                         # Will contain trained models
└── notebooks/                      # For experiments
```

## Quick Commands

### Option 1: Fast Test (2 min)
```bash
# Test all components without downloading Jina CLIP
python test_components.py
```

### Option 2: Full Pipeline (5-10 min)
```bash
# Install deps
pip install -r requirements.txt

# Run full training
python main.py
```

This will:
1. Download Jina CLIP v2 (~800MB, one-time)
2. Encode sample ads
3. Generate 10K synthetic clicks (80% train, 20% test)
4. Train user embeddings (50 epochs)
5. Train projector (30 epochs)
6. Precompute projected ad embeddings
7. **Evaluate performance with Precision@10, t-SNE plots, and heatmaps**
8. Save everything to `models/`, `data/`, and `evaluation_results/`

### Option 3: Individual Components
```bash
cd src

# Test each module
python projector.py
python click_data.py
python train_user_embeddings.py
python train_projector.py
```

## Swapping to Real Data

**Super easy—just one line change!**

In `main.py`, replace:
```python
click_gen = SyntheticClickGenerator(...)
```

With:
```python
from click_data import RealClickData
click_gen = RealClickData(data_path="your_data.csv")
```

Your CSV needs: `user_id, ad_id, clicked, position`

## Adding Real Ads with Images

In `main.py`, replace the `sample_ads` section with:
```python
import pandas as pd
from PIL import Image

ads_df = pd.read_csv("ads.csv")  # columns: ad_id, text, image_path

ad_embeddings_raw = {}
for _, row in ads_df.iterrows():
    img = Image.open(row['image_path']) if pd.notna(row['image_path']) else None
    z_ad = encoder.encode(text=row['text'], image=img)
    ad_embeddings_raw[row['ad_id']] = z_ad
```

## Key Design Decisions

1. **Jina CLIP v2 (frozen)** → Fast, small, good quality, no training needed
2. **Global prior for new users** → Simple, no metadata required
3. **InfoNCE losses** → State-of-art contrastive learning
4. **Modular click data** → Easy to swap synthetic ↔ real
5. **No user history stored** → Privacy-first by design

## What's Next (Phase 2)

- [ ] Thompson Sampling with position bias
- [ ] Serving API (rank ads per user)
- [ ] Online feedback loop (EMA updates)
- [ ] Evaluation metrics (Recall@K, NDCG)
- [ ] FAISS index for fast retrieval
- [ ] Demo UI

## Architecture Diagram

```
┌─────────────────┐
│ Ad Text + Image │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Jina CLIP v2   │  ← Frozen, 768-dim
│    (frozen)     │
└────────┬────────┘
         │ z_ad
         ▼
┌─────────────────┐
│   Projector     │  ← Trained, 768→128
│   (MLP 2-layer) │
└────────┬────────┘
         │ p_ad
         ▼
    User Space (128-dim)
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────┐
│ User Embeddings │   │   Scoring    │
│ (from clicks)   │   │ cos(u, p_ad) │
└─────────────────┘   └──────────────┘
```

## Privacy Features

✅ No demographics  
✅ No behavioral logs  
✅ Only abstract embeddings  
✅ GDPR-compliant deletion  
✅ Optional DP-SGD noise  

## Troubleshooting

**"No module named transformers"**
→ Run: `pip install -r requirements.txt`

**"CUDA out of memory"**
→ Main script auto-falls back to CPU, or reduce batch sizes in configs

**"No co-click edges"**
→ Increase `N_CLICKS` or lower `min_shared` threshold

**Jina CLIP download slow?**
→ One-time download, ~800MB. Use fast connection or wait patiently!

## Performance Expectations

**CPU (M1/M2 Mac or modern Intel):**
- Component tests: ~30 seconds
- Full pipeline: ~8-12 minutes

**GPU (any CUDA):**
- Component tests: ~15 seconds
- Full pipeline: ~2-3 minutes

## Team Division (Suggested)

- **Person 1:** Get main.py running, tune hyperparameters
- **Person 2:** Prepare real ad data (CSV + images)
- **Person 3:** Start Phase 2 (Thompson Sampling)
- **Person 4:** Build demo UI / evaluation metrics

## Questions?

Check `README.md` for full docs, or dive into source code—it's well commented!

---

**Good luck at the hackathon! 🚀**
