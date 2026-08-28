"""
eval/run_classifier.py

Day 5: does the question-type classifier actually work, and does routing on its
PREDICTIONS (not on oracle labels) improve retrieval?

The distinction matters. Earlier, oracle-label routing showed +0.031 at k=10.
A real classifier is worse than an oracle, so the honest question is whether
anything survives once its mistakes are included. If not, routing should be
dropped rather than shipped on the strength of a number that assumed perfect
labels.

Run:  python eval/run_classifier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus import load_corpus                      # noqa: E402
from eval.gold import load_gold                          # noqa: E402
from eval.harness import evaluate_retrieval              # noqa: E402
from pipeline.classify import (DEFAULT_THRESHOLD, classify,  # noqa: E402
                               evidence_k)
from retrieval.hybrid import HybridRetriever             # noqa: E402
from retrieval.rerank import MS_MARCO, RerankRetriever   # noqa: E402


def confusion(gold, threshold=DEFAULT_THRESHOLD):
    tp = fp = tn = fn = 0
    errors = []
    for q in gold:
        pred = classify(q.question, threshold).is_reasoning
        truth = q.is_reasoning
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
            errors.append(("false reasoning", q))
        elif not pred and truth:
            fn += 1
            errors.append(("missed reasoning", q))
        else:
            tn += 1
    return tp, fp, tn, fn, errors


def report_accuracy(gold):
    print("=" * 78)
    print("CLASSIFIER ACCURACY (question text only)")
    print("=" * 78)
    tp, fp, tn, fn, errors = confusion(gold)
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    majority = max(tp + fn, tn + fp) / n

    print(f"  accuracy   : {acc:.3f}   (always-'extractive' baseline: {majority:.3f})")
    print(f"  precision  : {prec:.3f}   of those called reasoning, how many are")
    print(f"  recall     : {rec:.3f}   of true reasoning, how many we caught")
    print(f"  F1         : {f1:.3f}")
    print(f"\n  confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    if acc <= majority:
        print("\n  WARNING: no better than always guessing the majority class.")
    return errors


def threshold_sweep(gold):
    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP")
    print("=" * 78)
    print(f"  {'thr':>5} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        tp, fp, tn, fn, _ = confusion(gold, t)
        n = tp + fp + tn + fn
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  {t:>5} {(tp+tn)/n:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print("\n  Note: this sweep is on the SAME data the rules were written from.\n"
          "  Treat it as a sanity check, not as evidence of generalization.")


def cross_validate(gold, folds: int = 5):
    """The rules were written after looking at these questions, so in-sample
    accuracy is optimistic. We cannot re-fit hand-written rules per fold, but
    we CAN re-fit the threshold per fold -- which is the only tuned parameter
    -- and see how much accuracy drops when it is chosen without seeing the
    test half."""
    import random
    print("\n" + "=" * 78)
    print(f"{folds}-FOLD CROSS-VALIDATION OF THE THRESHOLD")
    print("=" * 78)
    qs = list(gold)
    random.Random(0).shuffle(qs)
    size = len(qs) // folds
    accs = []
    for i in range(folds):
        test = qs[i * size:(i + 1) * size] if i < folds - 1 else qs[i * size:]
        train = [q for q in qs if q not in test]
        best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
        for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            tp, fp, tn, fn, _ = confusion(train, t)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        tp, fp, tn, fn, _ = confusion(test, best_t)
        acc = (tp + tn) / len(test)
        accs.append(acc)
        print(f"  fold {i+1}: threshold={best_t:>4} -> held-out accuracy={acc:.3f}")
    print(f"\n  mean held-out accuracy: {sum(accs)/len(accs):.3f}")


def show_errors(errors, limit=6):
    print("\n" + "=" * 78)
    print("WHERE IT GETS THINGS WRONG")
    print("=" * 78)
    for kind, q in errors[:limit]:
        c = classify(q.question)
        print(f"  [{kind}] score={c.score:+.1f}")
        print(f"     {q.question[:110]}")
        print(f"     fired: {', '.join(f.split(' ')[0] for f in c.fired) or '(nothing)'}")
    print(f"\n  total errors: {len(errors)}")


class WeightRouted:
    """Routes the FUSION WEIGHT per question. Kept only to demonstrate that it
    does not work -- deleting it would erase the evidence."""

    def __init__(self):
        self.name = "weight-routed (REJECTED)"
        self._variants = {}

    def _get(self, w):
        if w not in self._variants:
            h = HybridRetriever(method="weighted", w_bm25=w, candidate_pool=50)
            self._variants[w] = RerankRetriever(model_name=MS_MARCO, depth=20, base=h)
        return self._variants[w]

    def index(self, doc_name, chunks):
        for w in (0.25, 8.0):
            self._get(w).index(doc_name, chunks)

    def search(self, doc_name, query, k):
        c = classify(query)
        return self._get(8.0 if c.is_reasoning else 0.25).search(doc_name, query, k)


class KRouted:
    """Routes only the EVIDENCE-SET SIZE per question. This is the one that
    works: equal fusion weights everywhere, a bigger K for reasoning."""

    def __init__(self):
        self.name = "K-routed (ADOPTED)"
        self.r = RerankRetriever(
            model_name=MS_MARCO, depth=20,
            base=HybridRetriever(method="weighted", candidate_pool=50))

    def index(self, doc_name, chunks):
        self.r.index(doc_name, chunks)

    def search(self, doc_name, query, k):
        return self.r.search(doc_name, query, evidence_k(classify(query), k))


def routing_effect(gold, corpus):
    print("\n" + "=" * 78)
    print("DOES ROUTING ON PREDICTED LABELS HELP? (k=10)")
    print("=" * 78)
    fixed = RerankRetriever(model_name=MS_MARCO, depth=20, name="fixed k=10 (no routing)")
    for r in (fixed, WeightRouted(), KRouted()):
        rep = evaluate_retrieval(r, gold=gold, k=10, corpus=corpus)
        t = {n: sum(x.page_hit for x in s) / len(s)
             for n, s in rep.split().items() if s}
        avg = sum(len(x.retrieved_pages) for x in rep.results) / len(rep.results)
        print(f"  {r.name:<28} page={rep.page_recall:.3f} evid={rep.evidence_recall:.3f} "
              f"allpg={rep.all_pages_recall:.3f} avg_chunks={avg:.1f}")
        print(f"  {'':<28} " + "  ".join(f"{a}={b:.3f}" for a, b in t.items()))
    print(
        "\n  WEIGHT routing loses to doing nothing -- the classifier's mistakes\n"
        "  cost more than its correct calls gain, and the reranker already\n"
        "  removed the per-type weight conflict that motivated it.\n"
        "\n  K routing wins, for about 2 extra chunks on average. Same fusion\n"
        "  weight everywhere; reasoning questions simply get a bigger evidence\n"
        "  budget because they genuinely need more pages."
    )


def main():
    gold = load_gold()
    corpus = load_corpus({q.doc_name for q in gold})
    print(f"gold: {len(gold)} questions "
          f"({sum(q.is_reasoning for q in gold)} reasoning)\n")

    errors = report_accuracy(gold)
    threshold_sweep(gold)
    cross_validate(gold)
    show_errors(errors)
    routing_effect(gold, corpus)


if __name__ == "__main__":
    main()
