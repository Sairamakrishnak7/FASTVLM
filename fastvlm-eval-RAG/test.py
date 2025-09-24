import evaluate
import sys
import torch
import clip
# Append path to local clipscore_utils.py
sys.path.append(r"/Users/yashwanthkomaravolu/Desktop/Yashwanth/FastVLM")

from clipscore_utils import get_clip_score


def main():
    # Load other evaluation metrics
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Dummy captions
    references = ["A man is standing on a beach"]
    hypotheses = ["A person stands on a sandy shore"]

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # Compute CLIPScore -- Use this when we have an image
    # clip_score_value, _, _ = get_clip_score(
    # model=model,
    # images=references,
    # candidates=hypotheses,
    # device=device,
    # w=2.5)


    from clipscore_utils import get_refonlyclipscore, extract_all_captions

    # Extract features
    ref_caps = [[r] for r in references]  # nested list required
    hyp_caps = hypotheses

    hyp_feats = extract_all_captions(hyp_caps, model, device, batch_size=32, num_workers=0)
    clip_score_value, _ = get_refonlyclipscore(model, ref_caps, hyp_feats, device)
    print(f"RefCLIPScore (text-only): {clip_score_value:.4f}")


    print(f"CLIPScore: {clip_score_value:.4f}")

if __name__ == "__main__":
    main()
