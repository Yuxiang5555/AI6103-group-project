"""
pipeline.py -- end-to-end inference: text in -> UniMC baseline answer + improved answer.
For the Gradio demo (member F) and for spot checks.

    from pipeline import UniMCPlus
    p = UniMCPlus("IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English", "runs/tf_int/seed0/best.pt")
    p.predict([{"texta": "it 's just incredibly dull .", "textb": "",
                "question": "What is sentiment of follow review?",
                "choice": ["it's terrible", "it's great"]}])
    -> [{"choice": [...], "baseline": {"pred": 0, "probs": [...], "raw_logits": [...]},
         "improved": {"pred": 0, "probs": [...], "logits": [...]}}]

    python pipeline.py --ckpt runs/tf_int/seed0/best.pt      # runs the example above
"""
import argparse, json, torch
from unimc_core import load_unimc, official_probs
from decision_layers import DecisionScorer


class UniMCPlus:
    def __init__(self, model_path, ckpt_path=None, device=None, dtype=torch.bfloat16, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            dtype = torch.float32
        self.tok, self.enc, self.model = load_unimc(model_path, self.device, dtype, max_length)
        self.scorer = DecisionScorer.load(ckpt_path, self.device) if ckpt_path else None

    @torch.no_grad()
    def features(self, samples):
        batch = self.enc.collate([self.enc.encode(s) for s in samples])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        rep, raw = self.model(**batch)
        return rep, raw, batch["option_mask"]

    @torch.no_grad()
    def predict(self, samples):
        rep, raw, mask = self.features(samples)
        pb = official_probs(raw, mask)
        out = []
        new = self.scorer.score(rep, raw, mask) if self.scorer else None
        for i, s in enumerate(samples):
            m = len(s["choice"])
            r = {"choice": s["choice"],
                 "baseline": {"pred": int(pb[i].argmax()), "probs": pb[i, :m].tolist(), "raw_logits": raw[i, :m].tolist()}}
            if new is not None:
                pn = new[i].softmax(-1)
                r["improved"] = {"pred": int(pn.argmax()), "probs": pn[:m].tolist(), "logits": new[i, :m].tolist()}
            out.append(r)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    p = UniMCPlus(a.model, a.ckpt)
    ex = [{"texta": "it 's just incredibly dull .", "textb": "", "question": "What is sentiment of follow review?",
           "choice": ["it's terrible", "it's great"]},
          {"texta": "", "textb": "", "question": "Which of these is a mammal?",
           "choice": ["shark", "dolphin", "trout", "octopus"]}]
    print(json.dumps(p.predict(ex), indent=1))
