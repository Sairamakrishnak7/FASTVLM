import pandas as pd
import sys
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import clip
import torch
import evaluate

# Append path to local clipscore_utils.py
sys.path.append(r"/Users/yashwanthkomaravolu/Desktop/Yashwanth/FastVLM")
from clipscore_utils import get_refonlyclipscore

# ==== Device selection (MPS → CUDA → CPU) ====
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🚀 Using device: {device}")

# ==== Load captions ====
fastvlm_df = pd.read_csv(
    r"/Users/yashwanthkomaravolu/Desktop/Yashwanth/FastVLM/fastvlm-eval-RAG/fastvlm_captions.csv"
)
fastvlm_captions = fastvlm_df["Caption"].tolist()

gt_df = pd.read_csv(
    r"/Users/yashwanthkomaravolu/Desktop/Yashwanth/FastVLM/fastvlm-eval-RAG/gemini_gt.csv"
)
gt_captions = gt_df["Caption"].tolist()

# ==== Load FAISS index + metadata ====
index = faiss.read_index("kb_index.faiss")
with open("kb_metadata.pkl", "rb") as f:
    kb_data = pickle.load(f)
kb_captions = kb_data["captions"]

# ==== Embed FastVLM captions ====
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
query_embeddings = embed_model.encode(fastvlm_captions, convert_to_numpy=True)

# ==== Retrieve top-1 from KB ====
_, I = index.search(query_embeddings, k=1)
retrieved_captions = [kb_captions[idx[0]] for idx in I]

# ==== Load CLIP model ====
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model = clip_model.float().eval()  # force float32 for stability

# ==== RefCLIPScore ====
refclip_before, _ = get_refonlyclipscore(
    clip_model, [[gt] for gt in gt_captions], fastvlm_captions, device
)
refclip_after, _ = get_refonlyclipscore(
    clip_model, [[ref] for ref in retrieved_captions], fastvlm_captions, device
)
refclip_kb, _ = get_refonlyclipscore(
    clip_model, [[gt] for gt in gt_captions], retrieved_captions, device
)

# ==== ROUGE-L ====
rouge = evaluate.load("rouge")
rouge_before = rouge.compute(
    predictions=fastvlm_captions, references=gt_captions, use_stemmer=True
)["rougeL"]
rouge_after = rouge.compute(
    predictions=fastvlm_captions, references=retrieved_captions, use_stemmer=True
)["rougeL"]
rouge_kb = rouge.compute(
    predictions=retrieved_captions, references=gt_captions, use_stemmer=True
)["rougeL"]

# ==== METEOR ====
meteor = evaluate.load("meteor")
meteor_before = meteor.compute(
    predictions=fastvlm_captions, references=gt_captions
)["meteor"]
meteor_after = meteor.compute(
    predictions=fastvlm_captions, references=retrieved_captions
)["meteor"]
meteor_kb = meteor.compute(
    predictions=retrieved_captions, references=gt_captions
)["meteor"]

# ==== Print results in table ====
print("\n📊 Evaluation Summary:")
print(
    f"{'Metric':<15} {'Before RAG (FastVLM vs GT)':<30} {'After RAG (FastVLM vs KB)':<30} {'Retriever (KB vs GT)':<25}"
)
print("-" * 105)
print(
    f"{'RefCLIPScore':<15} {refclip_before:<30.4f} {refclip_after:<30.4f} {refclip_kb:<25.4f}"
)
print(
    f"{'ROUGE-L':<15} {rouge_before:<30.4f} {rouge_after:<30.4f} {rouge_kb:<25.4f}"
)
print(
    f"{'METEOR':<15} {meteor_before:<30.4f} {meteor_after:<30.4f} {meteor_kb:<25.4f}"
)
