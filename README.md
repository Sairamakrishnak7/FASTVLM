# FASTVLM
FastVLM: Lightweight Vision-Language Models for Video Understanding and Captioning Implementation and evaluation of Apple’s FastVLM models for frame extraction, caption generation, retrieval-augmented evaluation, and healthcare-oriented video summarization.

Installation Instructions for Evaluation:
# Create a fresh conda environment
conda create -n fastvlm_eval python=3.10 -y
conda activate fastvlm_eval

# Install all dependencies
pip install -r requirements.txt

# (Optional) Install Jupyter support
pip install notebook ipykernel
python -m ipykernel install --user --name=fastvlm_eval

