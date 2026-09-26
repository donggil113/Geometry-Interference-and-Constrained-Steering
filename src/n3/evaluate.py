"""Generation + utility evaluation of steering conditions, and judge scoring.

A *condition* = (setting, concept, method, alpha_mult, hyper-parameters).  For each condition we store
  * continuations for a fixed prompt set, sampled with a common random seed (common random numbers:
    every condition of a prompt batch sees the same sampling noise stream),
  * encoder self-consistency metrics accumulated over all edited positions during generation
    (target gain, protected-set change, new activations)  -> 'internal' (NOT behavior),
  * utility on held-out OWT text: delta CE and KL(clean || edited) of next-token predictions,
  * runtime of the edit computation (ms / position) and wall-clock generation time.
Behavioral scores are computed afterwards by independent judges (score_rows).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .features import diffmean_dir, random_dir, select_features
from .methods import EditSpec, Editor


@dataclass
class Condition:
    setting: str
    cid: str
    method: str
    alpha_mult: float
    K: int = 1
    protect_topm: int | None = None
    ridge_rel: float = 1.0
    fs_rounds: int = 3
    opt_mu_p: float = 1.0
    opt_mu_new: float = 1.0
    max_density: float = 0.05

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def build_editor(sae, stats, cond: Condition, collect=True) -> Editor:
    T, wT, _ = select_features(stats, cond.cid, cond.K, max_density=cond.max_density)
    alpha = cond.alpha_mult * stats["resid_norm_median"]
    gen = None
    if cond.method == "diffmean":
        gen = diffmean_dir(stats, cond.cid)
    elif cond.method == "random":
        gen = random_dir(sae.d, cond.cid)
    elif cond.method == "dec_rand_feat":
        # decoder row of a random feature with the same density band as typical selected features
        g = torch.Generator().manual_seed(777 + sum(map(ord, cond.cid)))
        pool = torch.where((stats["density"] > 1e-3) & (stats["density"] <= cond.max_density))[0]
        j = pool[torch.randint(0, len(pool), (1,), generator=g)]
        gen = sae.W_dec[j[0]] / sae.W_dec[j[0]].norm()
    spec = EditSpec(method=cond.method, T=T, wT=wT, alpha=alpha, protect_topm=cond.protect_topm,
                    ridge_rel=cond.ridge_rel, fs_rounds=cond.fs_rounds, opt_mu_p=cond.opt_mu_p,
                    opt_mu_new=cond.opt_mu_new, generic_dir=gen, feat_scale=stats["scale"])
    return Editor(sae, spec, collect=collect)


@torch.no_grad()
def utility(model, editor, util_tokens, down=None) -> dict:
    """Held-out OWT text with the edit applied at every non-BOS position.
    down = (cap_layer, downstream SAEWrap): independent internal readout at a later layer with a different
    SAE -- relative drift of features active in the clean run and number of newly active features."""
    if down is None:
        clean = model.logits(util_tokens)
        edited = model.logits(util_tokens, editor=editor)
    else:
        clean, hc = model.logits_capture(util_tokens, None, down[0])
        edited, he = model.logits_capture(util_tokens, editor, down[0])
    lp_c = torch.log_softmax(clean[:, :-1].float(), -1)
    lp_e = torch.log_softmax(edited[:, :-1].float(), -1)
    tgt = util_tokens[:, 1:].unsqueeze(-1)
    ce_c = -lp_c.gather(-1, tgt).mean()
    ce_e = -lp_e.gather(-1, tgt).mean()
    kl = (lp_c.exp() * (lp_c - lp_e)).sum(-1).mean()
    out = dict(ce_clean=float(ce_c), dce=float(ce_e - ce_c), kl=float(kl))
    if down is not None:
        sae2 = down[1]
        d = hc.shape[-1]
        fc = sae2.encode(hc[:, 1:].reshape(-1, d).float())
        fe = sae2.encode(he[:, 1:].reshape(-1, d).float())
        P = fc > 0
        out["down_p_rel_drift"] = float((((fe - fc) * P).norm(dim=-1) / (fc * P).norm(dim=-1).clamp_min(1e-6)).mean())
        out["down_p_lost_per_pos"] = float((P & (fe <= 0)).sum(-1).float().mean())
        out["down_new_active_per_pos"] = float(((fe > 0) & ~P).sum(-1).float().mean())
    return out


PROMPT_CUE = "This is a story about {}."


def prompt_tokens_with_cue(model, prompt_tokens, cid):
    """Prompting reference (not an activation edit): [BOS] + cue + original prompt tokens."""
    from .concepts import concept_by_id

    name = concept_by_id(cid)["name"].lower().replace("&", "and")
    cue = model.tokenizer(PROMPT_CUE.format(name) + "\n")["input_ids"]
    b = prompt_tokens.shape[0]
    return torch.cat([prompt_tokens[:, :1], torch.tensor(cue).expand(b, -1), prompt_tokens[:, 1:]], 1)


def run_condition(model, sae, stats, cond: Condition, prompt_tokens, util_tokens, max_new_tokens=40, seed=0,
                  down=None) -> dict:
    unedited = cond.method in ("none", "prompt")
    ed = build_editor(sae, stats, cond if not unedited else Condition(cond.setting, cond.cid, "none", 0.0))
    if cond.method == "prompt":
        prompt_tokens = prompt_tokens_with_cue(model, prompt_tokens, cond.cid)
    t0 = time.perf_counter()
    out = model.generate(prompt_tokens, None if unedited else ed, max_new_tokens=max_new_tokens, seed=seed)
    gen_secs = time.perf_counter() - t0
    internal = ed.summary()
    if unedited:
        internal = {k: 0.0 for k in internal}
    L = prompt_tokens.shape[1]
    conts = [model.tokenizer.decode(out[i, L:].tolist()) for i in range(out.shape[0])]
    cont_ids = [out[i, L:].tolist() for i in range(out.shape[0])]
    ued = build_editor(sae, stats, cond, collect=False) if not unedited else None
    util = utility(model, ued, util_tokens, down=down)
    return dict(key=cond.key(), cond=asdict(cond), conts=conts, cont_ids=cont_ids, internal=internal, utility=util,
                gen_secs=gen_secs, T=select_features(stats, cond.cid, cond.K, cond.max_density)[0].tolist())


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(l)["key"] for l in f if l.strip()}


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------ judging
def score_rows(rows, prompts, judges, scored_path: Path, use_nli=False):
    """Adds per-prompt judge scores to rows; caches by key in scored_path (jsonl)."""
    have = {}
    if scored_path.exists():
        for r in load_rows(scored_path):
            have[r["key"]] = r
    from .concepts import TOPICS, EMOTIONS

    todo = [r for r in rows if r["key"] not in have or (use_nli and "nli" not in have[r["key"]]["scores"])]
    for r in todo:
        conts = r["conts"]
        sc = dict(have[r["key"]]["scores"]) if r["key"] in have else {}
        if "topic_probs" not in sc:
            sc["topic_probs"] = judges["topic"].probs(conts).tolist()
            sc["emotion_probs"] = judges["emotion"].probs(conts).tolist()
            sc["nll"] = judges["fluency"].cond_nll(prompts, conts).tolist()
            sc["rel"] = judges["relevance"].cos(prompts, conts).tolist()
            from .judges import distinct_n

            sc["distinct2"] = [distinct_n(ids, 2) for ids in r["cont_ids"]]
        if use_nli and "nli" not in sc:
            fam, lab = r["cond"]["cid"].split(":")
            name = TOPICS[int(lab)] if fam == "topic" else EMOTIONS[int(lab)]
            sc["nli"] = judges["nli"].entail_prob(conts, fam, name).tolist()
        have[r["key"]] = dict(key=r["key"], scores=sc)
        with open(scored_path, "a") as f:
            f.write(json.dumps(have[r["key"]]) + "\n")
    return {k: v["scores"] for k, v in have.items()}


def load_judges(nli=False):
    from .judges import EmotionJudge, FluencyJudge, NLIJudge, RelevanceJudge, TopicJudge

    j = dict(topic=TopicJudge(), emotion=EmotionJudge(), fluency=FluencyJudge(), relevance=RelevanceJudge())
    if nli:
        j["nli"] = NLIJudge()
    return j
