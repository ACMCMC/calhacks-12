
# PrivAds: Privacy-First AI Ad Recommendation Engine

## Why PrivAds?
Big tech ad networks rely on tracking and personal data to target users. PrivAds is a new kind of ad platform: it learns what users like—without ever storing their identity, browsing history, or demographics. We use only click feedback and page context to serve relevant, privacy-respecting ads.

## What Makes It Unique
- **No user tracking or segments:** We use user embeddings, not personas or segments that could be traced back.
- **Multimodal ad understanding:** Ads are processed with a vision-language model (Jina CLIP v2) to extract both visual and textual signals.
- **Context-aware serving:** Ad selection considers both user embedding and the current page context.
- **Custom and dynamic ads:** The system can generate or enhance ads on the fly, tailored to user interests and page content.

## Technical Architecture
```
Ad (text + image) → Jina CLIP v2 (frozen) → z_ad (2048D) → Projector (MLP) → p_ad (512D)
                                                        ↓
User clicks → Co-click graph → Margin/Contrastive Loss → user_embeddings (512D)
                                                        ↓
Scoring: cos(user_emb, p_ad)
```
- **Frozen VLM:** Jina CLIP v2 is never fine-tuned, ensuring robust, general ad representations.
- **Learned Projector:** A 2-layer MLP maps ad embeddings into the user space.
- **User Embeddings:** Learned via margin-based contrastive loss on a co-click graph (users who clicked the same ads).
- **All embeddings L2-normalized** for cosine similarity.

## Quick Start
1. `pip install -r requirements.txt`
2. `python main.py`  
   - Loads Jina CLIP v2
   - Generates synthetic or loads real click data
   - Trains user embeddings and projector
   - Saves models and embeddings
3. Outputs in `models/` and `data/` (see below)

## Outputs
```
models/
  ├── user_embeddings.npy      # (n_users, 512)
  ├── global_mean.npy          # (512,)
  └── projector.pt             # Projector weights
data/
  ├── ad_embeddings_raw.npz    # (n_ads, 2048)
  └── ad_projected.npz         # (n_ads, 512)
```

## Technical Details
- **Contrastive Learning:** Margin-based loss encourages user embeddings to be closer to ads they clicked than to negatives, by a margin.
- **Co-click Graph:** Users are connected if they clicked the same ad; this graph is the basis for contrastive training.
- **Synthetic & Real Data:** Swap between synthetic and real click data with a single line of code.
- **Evaluation:** Metrics include Recall@100 and AUC for retrieval quality.
- **No PII, no history:** Only abstract vectors are stored; no user or behavioral data is ever saved.

## Component Testing
```bash
cd src
python ad_encoder.py         # Test ad encoder
python projector.py          # Test projector
python click_data.py         # Test click data
python train_user_embeddings.py
python train_projector.py
```

## Privacy by Design
- No user metadata, no tracking, no segments
- Embeddings are abstract and cannot be reversed to user data
- GDPR-compliant: delete a user by removing their embedding

## Next Steps
- [ ] Thompson Sampling for exploration
- [ ] Real-time serving API
- [ ] Feedback loop for online learning
- [ ] Real ad data with images/videos
- [ ] Advanced evaluation (NDCG, etc.)
- [ ] Fast ANN retrieval (FAISS)

## Demo
[https://privads-demo.onrender.com/](https://privads-demo.onrender.com/)

## License
Apache 2.0
