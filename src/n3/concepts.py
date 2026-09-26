"""Concepts, concept texts, prompts and the dev / held-out splits.

Concept families (each has an *independent* supervised judge, see judges.py):
  topic   : 10 Yahoo! Answers topics      (judge: fabriceyhc/bert-base-uncased-yahoo_answers_topics)
  emotion : 6 dair-ai/emotion labels      (judge: bhadresh-savani/distilbert-base-uncased-emotion)
  entity  : DBpedia-14 entity types       (judge: fabriceyhc/bert-base-uncased-dbpedia_14) -- held-out family (A1)

Splits are produced by a fixed seed *before* any steering result is seen and are written to
configs/splits.json.  Dev concepts / dev prompts are used for every selection decision (feature
selection hyper-parameters, protected-set rule, steering strength, method hyper-parameters).
Held-out concepts / held-out prompts are touched once, with the frozen protocol.
"""
from __future__ import annotations

import random

import numpy as np

from .data import ROOT, load_df, load_json, owt_texts, save_json

TOPICS = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
          "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ENTITIES = ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", "Mean Of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"]

SPLIT_SEED = 20260926
SPLITS_PATH = ROOT / "configs" / "splits.json"


def all_concepts():
    out = []
    for i, name in enumerate(TOPICS):
        out.append(dict(cid=f"topic:{i}", family="topic", label=i, name=name))
    for i, name in enumerate(EMOTIONS):
        out.append(dict(cid=f"emotion:{i}", family="emotion", label=i, name=name))
    for i, name in enumerate(ENTITIES):
        out.append(dict(cid=f"entity:{i}", family="entity", label=i, name=name))
    return out


def amend_splits_entity_family(n_test=8):
    """Amendment A1 (made before any held-out run, after the code review found 8 held-out concepts too few for
    concept-level inference): add a fully held-out *family* -- n_test DBpedia-14 entity types chosen by a fixed
    seed.  No entity concept is ever used for any dev decision."""
    sp = load_splits()
    if "entity" in sp["concepts"]:
        return sp
    rng = random.Random(SPLIT_SEED + 1)
    ids = [f"entity:{i}" for i in range(len(ENTITIES))]
    rng.shuffle(ids)
    sp["concepts"]["entity"] = dict(dev=[], test=sorted(ids[:n_test]))
    sp["test_concepts"] = sp["test_concepts"] + sp["concepts"]["entity"]["test"]
    sp.setdefault("amendments", []).append(
        "A1 (2026-09-26, before any held-out run): added held-out family 'entity' (DBpedia-14, judge "
        "fabriceyhc/bert-base-uncased-dbpedia_14) with 8 seeded classes to raise held-out concepts from 8 to 16.")
    save_json(sp, SPLITS_PATH)
    return sp


def make_splits(n_dev_prompts=24, n_test_prompts=48, prompt_tokens=12):
    """Deterministic dev/held-out split of concepts and prompts (GPT-2 tokenization for prompt length)."""
    from transformers import AutoTokenizer

    rng = random.Random(SPLIT_SEED)
    concepts = all_concepts()
    split = {}
    for fam, n_dev in [("topic", 5), ("emotion", 3)]:
        ids = [c["cid"] for c in concepts if c["family"] == fam]
        rng.shuffle(ids)
        split[fam] = dict(dev=sorted(ids[:n_dev]), test=sorted(ids[n_dev:]))
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    cands = []
    for t in owt_texts("prompts"):
        first = t.strip().split("\n")[0]
        ids = tok(first)["input_ids"]
        if len(ids) < prompt_tokens + 4:
            continue
        s = tok.decode(ids[:prompt_tokens])
        if "\n" in s or "http" in s or len(tok(s)["input_ids"]) != prompt_tokens:
            continue
        cands.append(s)
    rng.shuffle(cands)
    prompts = dict(dev=cands[:n_dev_prompts], test=cands[n_dev_prompts:n_dev_prompts + n_test_prompts])
    obj = dict(seed=SPLIT_SEED, concepts=split, prompts=prompts, prompt_tokens=prompt_tokens,
               dev_concepts=split["topic"]["dev"] + split["emotion"]["dev"],
               test_concepts=split["topic"]["test"] + split["emotion"]["test"])
    save_json(obj, SPLITS_PATH)
    return obj


def load_splits():
    return load_json(SPLITS_PATH)


def concept_by_id(cid):
    return next(c for c in all_concepts() if c["cid"] == cid)


def concept_texts(n_per_concept=300, seed=0):
    """Train-split texts per concept (used for feature selection and DiffMean only)."""
    rng = np.random.RandomState(seed)
    out = {}
    y = load_df("yahoo_train0")
    for i in range(10):
        sub = y[y["topic"] == i]
        sub = sub.iloc[rng.permutation(len(sub))[:n_per_concept]]
        texts = (sub["question_title"].fillna("") + " " + sub["best_answer"].fillna("")).str.replace("\\n", " ", regex=False)
        out[f"topic:{i}"] = [t[:600] for t in texts.tolist()]
    e = load_df("emotion_train")
    for i in range(6):
        sub = e[e["label"] == i]
        sub = sub.iloc[rng.permutation(len(sub))[:n_per_concept]]
        out[f"emotion:{i}"] = sub["text"].tolist()
    db = load_df("dbpedia_train")
    for i in range(len(ENTITIES)):
        sub = db[db["label"] == i]
        sub = sub.iloc[rng.permutation(len(sub))[:n_per_concept]]
        out[f"entity:{i}"] = [(t + ". " + c)[:600] for t, c in zip(sub["title"].fillna(""), sub["content"].fillna(""))]
    return out
