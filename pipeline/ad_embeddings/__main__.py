# Orchestrate the ad embedding pipeline
import subprocess
import sys
from pathlib import Path

def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Step failed: {desc}")
        sys.exit(1)

# 1. Generate ad embeddings if not present
if not Path("data/ad_embeddings_raw.npz").exists():
    run_step("python -m pipeline.ad_embeddings.generate_ad_embeddings", "Generating ad embeddings")
else:
    print("ad_embeddings_raw.npz already exists, skipping embedding generation.")

# 2. Build user co-click graph if not present
if not Path("data/user_graph.npz").exists():
    run_step("python -m pipeline.ad_embeddings.train_user_embeddings", "Building user co-click graph")
else:
    print("user_graph.npz already exists, skipping user graph.")

# 3. Train projector
run_step("python -m pipeline.ad_embeddings.train_projector", "Training projector")

print("\nPipeline complete.")
