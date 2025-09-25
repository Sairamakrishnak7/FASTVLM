# Agentic Video Understanding with RAG + SLMs
Video understanding in healthcare demands lightweight, real-time models for safety-critical insights.
Small Language Models (SLMs) struggle with temporal reasoning across video frames.
 Hallucinations in AI must be addressed to ensure clinical safety and reliability in healthcare applications

 ## Problem Statement:
 Design a multimodal video reasoning agent using open-source SLMs and RAG pipelines to enhance situational awareness in clinical workflows.
The agent combines vision + language + retrieval to support real-time, explainable, and lightweight deployments in healthcare environments



## Installation/Instructions for FastVLM :

### Datasets 
https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip 

### Enable conda environment 
conda create -n fastvlm python=3.10 -y <br>
conda activate fastvlm

### Git command  
git clone https://github.com/apple/ml-fastvlm.git <br>
cd ml-fastvlm

### Install Depedencies:
pip install -e . <br>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 <br>
python predict.py --model-path "checkpoints/llava-fastvithd_1.5b_stage3" --image-file "C:\Users\anmol\Downloads\testforapple\man.jpg" --prompt "Describe the image." <br>

## Installation Instructions for Evaluation:
### Create a fresh conda environment 
conda create -n fastvlm_eval python=3.10 -y <br>
conda activate fastvlm_eval

### Install all dependencies
pip install -r requirements.txt

### (Optional) Install Jupyter support
pip install notebook ipykernel <br>
python -m ipykernel install --user --name=fastvlm_eval




