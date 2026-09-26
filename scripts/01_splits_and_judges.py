"""Create the frozen dev/held-out splits and validate the independent judges.

Outputs configs/splits.json and results/validation/judge_validation.json.
"""
import numpy as np

import n3  # noqa: F401
from n3.concepts import make_splits
from n3.data import ROOT, load_df, save_json
from n3.judges import EmotionJudge, FluencyJudge, NLIJudge, RelevanceJudge, TopicJudge

rng = np.random.RandomState(0)
splits = make_splits()
print({k: v for k, v in splits.items() if k != "prompts"})
print("n prompts", {k: len(v) for k, v in splits["prompts"].items()}, splits["prompts"]["dev"][:3])

rep = {}
yt = load_df("yahoo_test")
yt = yt.iloc[rng.permutation(len(yt))[:600]]
texts = (yt["question_title"].fillna("") + " " + yt["best_answer"].fillna("")).tolist()
tj = TopicJudge()
pt = tj.probs(texts)
rep["topic_judge_acc_identity_map"] = float((pt.argmax(1) == yt["topic"].values).mean())
rep["topic_judge_confusion_diag"] = [float((pt.argmax(1)[yt["topic"].values == i] == i).mean()) for i in range(10)]

et = load_df("emotion_test")
et = et.iloc[rng.permutation(len(et))[:600]]
ej = EmotionJudge()
pe = ej.probs(et["text"].tolist())
rep["emotion_judge_acc"] = float((pe.argmax(1) == et["label"].values).mean())
rep["emotion_judge_per_class"] = [float((pe.argmax(1)[et["label"].values == i] == i).mean()) for i in range(6)]

nli = NLIJudge()
# NLI sanity: entailment prob for correct vs wrong topic on 60 yahoo texts
sub = yt.iloc[:60]
st = (sub["question_title"].fillna("") + " " + sub["best_answer"].fillna("")).tolist()
from n3.concepts import TOPICS
right = np.array([nli.entail_prob([t], "topic", TOPICS[l])[0] for t, l in zip(st, sub["topic"].values)])
wrong = np.array([nli.entail_prob([t], "topic", TOPICS[(l + 3) % 10])[0] for t, l in zip(st, sub["topic"].values)])
rep["nli_topic_right_mean"], rep["nli_topic_wrong_mean"] = float(right.mean()), float(wrong.mean())
rep["nli_topic_auc_right_vs_wrong"] = float((right[:, None] > wrong[None, :]).mean())

fj = FluencyJudge()
good = ["The weather was pleasant, so we walked to the park and had lunch by the river."]
bad = ["weather the the park lunch river river so so walked had had by pleasant the."]
rep["fluency_nll_good"], rep["fluency_nll_bad"] = float(fj.cond_nll(["Yesterday"], good)[0]), float(fj.cond_nll(["Yesterday"], bad)[0])
rj = RelevanceJudge()
rep["relevance_sanity"] = float(rj.cos(["The stock market fell sharply today."], ["Shares dropped as investors panicked."])[0]), float(rj.cos(["The stock market fell sharply today."], ["My cat likes to sleep on the sofa."])[0])
print(rep)
save_json(rep, ROOT / "results" / "validation" / "judge_validation.json")
