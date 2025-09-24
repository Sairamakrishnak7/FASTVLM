import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# === Step 1: Load KB CSV ===
kb_path = r"/Users/yashwanthkomaravolu/Desktop/Yashwanth/FastVLM/fastvlm-eval-RAG/gemini_kb.csv"
df = pd.read_csv(kb_path)
captions = df["Caption"].tolist()
image_names = df["Image_name"].tolist()

# === Step 2: Encode Captions ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # ~384-dim vectors
embeddings = model.encode(captions, convert_to_numpy=True)

# === Step 3: Build FAISS Index ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# === Step 4: Save FAISS Index & Metadata ===
faiss.write_index(index, "kb_index.faiss")

with open("kb_metadata.pkl", "wb") as f:
    pickle.dump({"captions": captions, "image_names": image_names}, f)

print(f"✅ Retriever built: {len(captions)} entries indexed.")
