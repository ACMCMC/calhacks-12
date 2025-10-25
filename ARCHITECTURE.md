# PrivAds Architecture: Multi-Modal Embedding Pipeline

## 🎯 System Overview

A privacy-preserving advertising system that learns user preferences from behavioral signals and matches them with semantically-rich ad embeddings.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Ad Text ──▶ [Ad Encoder] ──▶ Ad Embedding ──┐            │
│              (Fine-tuned LM)   (D_ad dims)     │            │
│                                                 │            │
│                                                 ▼            │
│                                          [Projector]         │
│                                          (D_ad → D_user)     │
│                                                 │            │
│  User ID ──▶ [User Embedding] ─────────────────┤            │
│              (Learned Lookup)                   │            │
│              (D_user dims)                      ▼            │
│                                            Similarity Score  │
│                                            (Cosine/Dot)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Component 1: Ad Encoder (Text → Embedding)

### Purpose
Transform textual ad features into dense semantic embeddings that capture ad characteristics.

### Architecture Options

#### Option A: Fine-tune Small Encoder (Recommended)
- **Base Model**: MiniLM, TinyBERT, or DistilBERT (< 50M params)
- **Input**: "for children, expensive, sustainable, outdoor gear, waterproof"
- **Output**: Fixed-size embedding (e.g., 384 or 768 dims)
- **Training**: 
  - Start with sentence-transformers pre-trained model
  - Fine-tune on ad similarity tasks (similar categories → close embeddings)
  
```python
from sentence_transformers import SentenceTransformer

# Base model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# Fine-tune on ad similarity pairs
# (ad_text_1, ad_text_2, similarity_score)
```

#### Option B: Custom Encoder
- BERT/RoBERTa backbone + [CLS] token pooling
- Additional MLP head for dimensionality control

### Training Strategy for Ad Encoder

1. **Pre-training Phase**: Use existing sentence-transformers
2. **Fine-tuning Phase**:
   - Create synthetic ad pairs from same/different categories
   - Use contrastive loss (SimCLR, InfoNCE)
   - Data augmentation: paraphrase, synonym replacement

### Considerations
- **Dimensionality**: D_ad = 256-768 (trade-off between expressiveness and compute)
- **Normalization**: L2-normalize embeddings for cosine similarity
- **Privacy**: Model processes ad text only, no user data

---

## 🧑 Component 2: User Embeddings (Behavioral Learning)

### Purpose
Learn user preference representations from click behavior without any textual features.

### Architecture

#### Embedding Table
```python
user_embeddings = nn.Embedding(
    num_embeddings=num_users,
    embedding_dim=D_user  # e.g., 128-256
)
```

### Training Strategy: Contrastive Learning

#### Triplet Loss Approach

**Data Structure**: `(user_id, ad_id, clicked: bool)`

**Triplet Formation**:
```
Anchor:   User A embedding
Positive: User B who clicked SAME ads as A
Negative: User C who clicked DIFFERENT ads from A
```

**Loss Function**:
```python
L_triplet = max(0, ||u_a - u_p||² - ||u_a - u_n||² + margin)
```

#### Implementation Strategies

**Strategy 1: Co-Click Based Triplets**
- **Positive Pairs**: Users who clicked the same ad
- **Negative Pairs**: Users who never clicked common ads
- **Mining**: Hard negative mining (users with overlapping but not identical clicks)

```python
# Pseudo-code
for batch in dataloader:
    anchor_user = get_user(batch)
    
    # Positive: another user who clicked same ad
    positive_user = sample_user_who_clicked(batch.ad_id)
    
    # Hard negative: user who clicked different ads in same category
    negative_user = sample_hard_negative(anchor_user, same_category=True)
    
    loss = triplet_loss(
        anchor=user_emb[anchor_user],
        positive=user_emb[positive_user],
        negative=user_emb[negative_user],
        margin=0.5
    )
```

**Strategy 2: Multiple Positives per Anchor**
- Use all users who clicked the same ad set as positives
- NT-Xent loss (InfoNCE) instead of triplet loss

```python
# Batch of users who clicked ad_i
users_for_ad_i = get_users_clicked(ad_id=i)
embeddings = user_emb[users_for_ad_i]

# NT-Xent: pull together all users who clicked same ad
loss = nt_xent_loss(embeddings, temperature=0.07)
```

**Strategy 3: Graph-Based Learning**
- Build user-ad bipartite graph
- Use Graph Neural Network (GNN) to propagate info
- Or use Random Walk with Restart for similarity

### Addressing Sparse Data

**Problem**: Many users have few clicks → sparse training signal

**Solutions**:
1. **Hierarchical Modeling**: User clusters → individual users
2. **Meta-Learning**: Learn initialization that adapts quickly to new users
3. **Regularization**: Strong L2 regularization to prevent overfitting
4. **Data Augmentation**: Probabilistic clicks, temporal sliding windows

---

## 🔄 Component 3: Projection Layer (Ad Space → User Space)

### Purpose
Map ad embeddings into the user embedding space to enable direct comparison.

### Why Project?

The ad encoder and user embeddings are trained separately with different objectives:
- **Ad space**: Semantic similarity (text-based)
- **User space**: Behavioral similarity (click-based)

These spaces may have different geometries and scales.

### Architecture Options

#### Option 1: Linear Projection (Simplest)
```python
projector = nn.Linear(D_ad, D_user)
```

#### Option 2: MLP Projection (More Expressive)
```python
projector = nn.Sequential(
    nn.Linear(D_ad, D_ad // 2),
    nn.ReLU(),
    nn.Linear(D_ad // 2, D_user)
)
```

#### Option 3: Attention-Based Projection
- Learn which ad features are most relevant for user matching

### Training Strategy

**Objective**: Align projected ad embeddings with user embeddings based on clicks.

**Loss Function**: Ranking Loss
```python
# For each (user, ad, clicked) sample

ad_emb_projected = projector(ad_encoder(ad_text))
user_emb = user_embeddings[user_id]

if clicked:
    # Pull together
    loss = 1 - cosine_similarity(ad_emb_projected, user_emb)
else:
    # Push apart (negative sample)
    loss = max(0, cosine_similarity(ad_emb_projected, user_emb) - margin)
```

**Alternative**: Use the click prediction as supervision
```python
# Logit = similarity between projected ad and user
logit = dot_product(ad_emb_projected, user_emb)
click_prob = sigmoid(logit)

loss = binary_cross_entropy(click_prob, clicked_label)
```

---

## 🔧 Training Pipeline

### Stage 1: Pre-train Ad Encoder (Optional but Recommended)
- Use sentence-transformers or similar
- Fine-tune on ad similarity data if available
- **Output**: Frozen or semi-frozen ad encoder

### Stage 2: Joint Training of User Embeddings + Projector

#### Approach A: Two-Phase Training
1. **Phase 1**: Train user embeddings with triplet loss (freeze projector)
2. **Phase 2**: Train projector to align spaces (freeze user embeddings)

#### Approach B: End-to-End Training (Recommended)
Train user embeddings and projector jointly:

```python
# Training loop
for batch in dataloader:
    user_ids, ad_ids, clicked = batch
    
    # Get ad embeddings (frozen or fine-tuned encoder)
    ad_texts = get_ad_texts(ad_ids)
    ad_embs = ad_encoder(ad_texts)
    ad_embs_projected = projector(ad_embs)
    
    # Get user embeddings
    user_embs = user_embedding_table(user_ids)
    
    # Loss 1: Click prediction (alignment)
    similarity = cosine_similarity(ad_embs_projected, user_embs)
    loss_click = bce_loss(similarity, clicked)
    
    # Loss 2: User contrastive loss (user space structure)
    loss_user_contrastive = compute_triplet_loss(user_embs, batch)
    
    # Combined loss
    loss = loss_click + λ * loss_user_contrastive
```

### Hyperparameters
- **Embedding Dimensions**: D_ad = 384, D_user = 256
- **Batch Size**: 256-1024 (need enough users per batch for contrastive)
- **Learning Rate**: 1e-4 for projector, 1e-3 for embeddings
- **Temperature**: 0.05-0.1 for contrastive losses
- **Margin**: 0.3-0.5 for triplet loss
- **λ**: 0.1-1.0 for balancing losses

---

## 🎲 Handling Cold Start

### New Users (No Click History)
**Problem**: Can't compute user embedding without behavioral data.

**Solutions**:
1. **Default Embedding**: Use mean user embedding or zero vector
2. **Content-Based Fallback**: If user provides any demographic/textual info, encode it
3. **Fast Adaptation**: After first few clicks, update user embedding with mini fine-tuning
4. **Contextual Bandits**: Explore diverse ads initially to learn preferences quickly

### New Ads (No Click History Yet)
**Advantage**: We have textual features! 
- Ad encoder generates embedding immediately
- Projector maps to user space
- Can rank against all users from day 1

---

## 📊 Evaluation Metrics

### Offline Metrics
1. **Ranking Metrics**: 
   - Precision@K, Recall@K
   - NDCG@K
   - MRR (Mean Reciprocal Rank)

2. **Embedding Quality**:
   - User embedding cluster coherence
   - Ad embedding semantic consistency

3. **Alignment Quality**:
   - Cosine similarity distribution for clicked vs non-clicked pairs

### Online Metrics (A/B Testing)
- CTR (Click-Through Rate)
- Conversion Rate
- User Engagement Time
- Revenue per User

---

## 🔐 Privacy Considerations

### Advantages
1. **No PII in Model**: User embeddings are learned IDs, not demographics
2. **Behavioral Only**: No sensitive textual data from users
3. **Federated Potential**: User embeddings could be computed on-device

### Concerns
1. **Embedding Inversion**: Can someone reverse-engineer user preferences from embeddings?
   - **Mitigation**: Add differential privacy noise to embeddings
2. **Linkage Attacks**: Can embeddings be linked across platforms?
   - **Mitigation**: Periodic re-initialization, embedding rotation

### Privacy-Preserving Extensions
- **Differential Privacy**: Add noise to user embedding updates
- **Federated Learning**: Users compute their own embeddings locally
- **Secure Multi-Party Computation**: Match users to ads without revealing embeddings

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up data pipeline for `(user_id, ad_id, clicked)`
- [ ] Implement ad encoder (start with pre-trained sentence-transformers)
- [ ] Create user embedding table
- [ ] Implement basic triplet loss for users

### Phase 2: Projection & Integration (Weeks 3-4)
- [ ] Implement projector network
- [ ] Joint training loop (user embeddings + projector)
- [ ] Hyperparameter tuning
- [ ] Evaluation pipeline (offline metrics)

### Phase 3: Optimization (Weeks 5-6)
- [ ] Hard negative mining strategies
- [ ] Model compression (quantization, distillation)
- [ ] Serving infrastructure (vector database for similarity search)
- [ ] Cold start strategies

### Phase 4: Privacy & Production (Weeks 7-8)
- [ ] Add differential privacy mechanisms
- [ ] A/B testing framework
- [ ] Monitoring & logging
- [ ] Documentation & deployment

---

## 🤔 Open Questions & Design Decisions

### Q1: Should we fine-tune the ad encoder during training?
**Option A**: Freeze ad encoder (faster, more stable)
**Option B**: Fine-tune end-to-end (potentially better alignment)

**Recommendation**: Start frozen, fine-tune later if needed.

### Q2: What's the optimal dimension ratio D_ad vs D_user?
**Trade-off**: 
- Larger D_ad: More semantic richness
- Larger D_user: More capacity to capture diverse user behaviors

**Recommendation**: Start with D_ad=384, D_user=256. Project down.

### Q3: How to handle implicit negative feedback?
**Problem**: We have `clicked=True` samples, but `clicked=False` could mean:
- User saw ad and disliked it (true negative)
- User never saw ad (unknown)

**Solutions**:
- Use only displayed but not clicked as negatives
- Sample hard negatives from similar categories

### Q4: Should projector be symmetric (can we go user → ad space)?
**Current**: Ad → User (one-way)
**Alternative**: Bidirectional projection with cycle consistency

**Recommendation**: Start one-way for simplicity.

---

## 📚 Related Work & Inspirations

### Academic Papers
1. **Two-Tower Models**: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (Google, 2019)
2. **Contrastive Learning**: "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
3. **User Embedding**: "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (Pinterest, 2018)

### Industry Implementations
- **YouTube**: Two-tower architecture for video recommendations
- **Spotify**: Playlist embeddings from collaborative filtering
- **Airbnb**: Listing embeddings from booking sessions

---

## 🛠️ Tech Stack Recommendations

### Core ML
- **Framework**: PyTorch (more flexible for custom loss functions)
- **Transformers**: Hugging Face `transformers` + `sentence-transformers`
- **Training**: PyTorch Lightning (cleaner training loops)

### Data & Serving
- **Data Pipeline**: PyArrow, DuckDB for efficient click data processing
- **Vector DB**: Faiss, Milvus, or Qdrant for similarity search
- **Serving**: FastAPI + ONNX Runtime for inference

### Monitoring
- **Experiment Tracking**: Weights & Biases or MLflow
- **Model Versioning**: DVC or MLflow
- **Metrics**: Prometheus + Grafana

---

## 💡 Next Steps

1. **Define Data Schema**: Finalize the structure of click data, ad features
2. **Baseline Model**: Implement simplest version (frozen ad encoder, linear projector)
3. **Evaluation Suite**: Build offline evaluation before complex training
4. **Iterate**: Start simple, add complexity based on error analysis

**Key Success Metric**: Can the system accurately predict which ads a user will click based solely on behavioral embeddings?

