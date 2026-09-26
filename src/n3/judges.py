"""Independent behavioral judges (none of them shares weights with GPT-2 or any SAE).

  TopicJudge    : BERT fine-tuned on Yahoo! Answers topics (10-way)
  EmotionJudge  : DistilBERT fine-tuned on dair-ai/emotion (6-way)
  NLIJudge      : DeBERTa-v3 zero-shot NLI (secondary concept judge, no dataset-specific training)
  FluencyJudge  : Qwen2.5-0.5B conditional NLL of the continuation given the prompt
  RelevanceJudge: all-MiniLM-L6-v2 cosine(prompt, continuation)
"""
from __future__ import annotations

import numpy as np
import torch

JUDGE_REVISIONS = {
    "fabriceyhc/bert-base-uncased-yahoo_answers_topics": "968176fc24a2eb73cac26ab4312d8b22da98486a",
    "bhadresh-savani/distilbert-base-uncased-emotion": "ce6f4ffcde7642ca2cac02381a16da38e5498ff7",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


class ClassifierJudge:
    def __init__(self, repo):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        rev = JUDGE_REVISIONS[repo]
        self.tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).eval()

    @torch.no_grad()
    def probs(self, texts, bs=64, max_length=128):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, self.model.config.num_labels))


class TopicJudge(ClassifierJudge):
    def __init__(self):
        super().__init__("fabriceyhc/bert-base-uncased-yahoo_answers_topics")


class EmotionJudge(ClassifierJudge):
    def __init__(self):
        super().__init__("bhadresh-savani/distilbert-base-uncased-emotion")


class NLIJudge:
    TEMPLATES = {"topic": "This text is about {}.", "emotion": "This text expresses {}."}

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        repo = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
        rev = JUDGE_REVISIONS[repo]
        self.tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = lab.get("entailment", 0)

    @torch.no_grad()
    def entail_prob(self, texts, family, label_name, bs=32):
        hyp = self.TEMPLATES[family].format(label_name.lower().replace("&", "and"))
        out = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = self.tok(chunk, [hyp] * len(chunk), return_tensors="pt", padding=True, truncation=True, max_length=192)
            logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, self.entail].numpy())
        return np.concatenate(out) if out else np.zeros(0)


class FluencyJudge:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = "Qwen/Qwen2.5-0.5B"
        rev = JUDGE_REVISIONS[repo]
        self.tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()

    @torch.no_grad()
    def cond_nll(self, prompts, conts, bs=16):
        """Mean per-token NLL (nats) of each continuation given its prompt."""
        res = []
        for i in range(0, len(prompts), bs):
            P, C = prompts[i:i + bs], conts[i:i + bs]
            ids, masks = [], []
            for p, c in zip(P, C):
                pi = self.tok(p)["input_ids"]
                ci = self.tok(c)["input_ids"]
                ids.append(pi + ci)
                masks.append([0] * len(pi) + [1] * len(ci))
            L = max(len(x) for x in ids)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
            cm = torch.tensor([m + [0] * (L - len(m)) for m in masks])
            logits = self.model(input_ids=inp, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = cm[:, 1:].float()
            res.append(((-lp * m).sum(-1) / m.sum(-1).clamp_min(1)).numpy())
        return np.concatenate(res) if res else np.zeros(0)


class RelevanceJudge:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        repo = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(repo, revision=JUDGE_REVISIONS[repo], device="cpu")

    def embed(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def cos(self, a, b):
        return (self.embed(a) * self.embed(b)).sum(-1)


def distinct_n(text_tokens, n=2):
    grams = [tuple(text_tokens[i:i + n]) for i in range(len(text_tokens) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)
