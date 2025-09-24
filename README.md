# FASTVLM
### FastVLM: Lightweight Vision-Language Models for Video Understanding and Captioning Implementation and evaluation of Apple’s FastVLM models for frame extraction, caption generation, retrieval-augmented evaluation, and healthcare-oriented video summarization.

# Installation/Instructions for FastVLM :

### Datasets 
https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip 

## Enable conda environment 
conda create -n fastvlm python=3.10 -y <br>
conda activate fastvlm

## Git command  
git clone https://github.com/apple/ml-fastvlm.git <br>
cd ml-fastvlm

## Install Depedencies:
pip install -e . <br>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 <br>
python predict.py --model-path "checkpoints/llava-fastvithd_1.5b_stage3" --image-file "C:\Users\anmol\Downloads\testforapple\man.jpg" --prompt "Describe the image." <br>

# Installation Instructions for Evaluation:
## Create a fresh conda environment 
conda create -n fastvlm_eval python=3.10 -y <br>
conda activate fastvlm_eval

## Install all dependencies
pip install -r requirements.txt

## (Optional) Install Jupyter support
pip install notebook ipykernel <br>
python -m ipykernel install --user --name=fastvlm_eval




