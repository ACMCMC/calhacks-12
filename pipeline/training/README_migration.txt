# Migration Plan: Canonical Training Directory

All core modules are being consolidated into `pipeline/training/`:
- click_data.py (keep best version, merge RealClickDataQuantile)
- ad_encoder.py (Jina CLIP v2 wrapper)
- projector.py (MLP projector)
- train_user_embeddings.py (joint user/projector training)
- train_projector.py (if needed, or merge into user training)
- evaluate.py (evaluation/validation)

All scripts and imports will reference `pipeline/training/` as the canonical location. After migration, `pipeline/ad_embeddings/` will be cleaned up.

**Next steps:**
1. Move/merge best code into `pipeline/training/`.
2. Update all imports and scripts.
3. Delete obsolete files from `pipeline/ad_embeddings/`.
4. Test full pipeline.
