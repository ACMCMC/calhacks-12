# Orchestrate the ad embedding pipeline
import subprocess

print("[1/3] Generating ad embeddings...")
subprocess.run(["python", "pipeline/ad_embeddings/generate_ad_embeddings.py"], check=True)

print("[2/3] Building user co-click graph...")
subprocess.run(["python", "pipeline/ad_embeddings/train_user_embeddings.py"], check=True)

print("[3/3] Training projector (stub)...")
subprocess.run(["python", "pipeline/ad_embeddings/train_projector.py"], check=True)

print("Pipeline complete.")
