"""The seam cross-encoder (design doc 8.3): P(page b's opening continues page a's ending).

A small pre-trained language model (for example all-MiniLM-L6-v2, Apache-2.0) read as a
cross-encoder: input "[CLS] a's last lines [SEP] b's first lines", one output. Fine-tuned on the
training documents' true neighbour pairs against wrong ones (the scan neighbour when it is wrong,
the most similar-looking page, a random page of the same document).

It is trained on one half of the training documents (the seam half); the evidence model of v8 / v9
then trains on the other half, so the seam's scores on its own training pages cannot inflate it.

Needs ``torch`` and ``transformers`` (requirements-seam.txt) and a local copy of the base model
(config.SEAM_BASE_MODEL_PATH); nothing is downloaded. Runs on the CPU only.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

META = "seam_meta.json"


def seam_half(names: Sequence[str]) -> set[str]:
    """The training documents the seam model learns from: a fixed half, by name."""
    import hashlib

    return {n for n in names if int(hashlib.md5(n.encode()).hexdigest(), 16) % 2 == 0}


class SeamScorer:
    def __init__(self, path: Path, max_length: Optional[int] = None, batch: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
        meta = Path(path) / META
        self.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        # score with the input length the model was trained with (shorter = faster on CPU)
        self.max_length = max_length or int(self.meta.get("max_length") or 256)
        self.batch = batch

    def score(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(left), self.batch):
                enc = self.tokenizer(list(left[i:i + self.batch]), list(right[i:i + self.batch]), truncation=True,
                                     max_length=self.max_length, padding=True, return_tensors="pt")
                logits = self.model(**enc).logits.reshape(-1)
                out.append(self.torch.sigmoid(logits).numpy())
        return np.concatenate(out) if out else np.zeros(0)


def page_texts(cf) -> dict[int, tuple[str, str]]:
    """page -> (its ending, its opening): the body's last and first lines (v2 features)."""
    return {p.page_number: (p.tail_text or "", p.head_text or "") for p in cf.pages}


def seam_for_pairs(scorer: Optional[SeamScorer], cf, pairs: Sequence[tuple[int, int]]) -> Optional[dict]:
    if scorer is None:
        return None
    texts = page_texts(cf)
    real = [(a, b) for a, b in pairs if a in texts and b in texts]
    probs = scorer.score([texts[a][0] for a, _ in real], [texts[b][1] for _, b in real])
    return {pair: float(p) for pair, p in zip(real, probs)}


NEGATIVE_KINDS = ("scan neighbour", "most similar", "random")
# What the seam learns from (2026-11-24): the wrong scan neighbour and two random pages of the same
# document. The "most similar" page is left out: with it, the seam learned only "the most similar
# page is not the next one" (AUC 0.50 against random pages, below TF-IDF's 0.70), which v8 / v9's
# TF-IDF features already say. seam_diagnose.py still measures it against all three kinds.
TRAIN_NEGATIVES = ("scan neighbour", "random", "random")


def training_pairs(examples: Iterable, rng: random.Random, negatives: Sequence[str] = TRAIN_NEGATIVES) -> list[tuple]:
    """(a's ending, b's opening, label, document, kind, a, b) from (chart, truth) examples; kind is
    "true next" or one of NEGATIVE_KINDS (the first that picked the page). ``negatives``: the wrong
    pages to draw for each true pair, by kind ("random" may repeat: distinct pages)."""
    from seqlib.v2.features import chart_features

    rows = []
    for chart, truth in examples:
        order = [p for p in truth.main_order if p in chart.by_number()]
        if len(order) < 2:
            continue
        cf = chart_features(chart, order)
        texts = page_texts(cf)
        scan = sorted(order)
        nxt_scan = {a: b for a, b in zip(scan, scan[1:])}
        true_next = dict(zip(order, order[1:]))
        idx = {p.page_number: i for i, p in enumerate(cf.pages)}
        for a, b in true_next.items():
            rows.append((texts[a][0], texts[b][1], 1, chart.name, "true next", a, b))
            wrong: dict[int, str] = {}
            if "scan neighbour" in negatives and nxt_scan.get(a) not in (None, b):
                wrong.setdefault(nxt_scan[a], "scan neighbour")
            if "most similar" in negatives:
                sims = cf.char_sim[idx[a]].copy()
                for j in np.argsort(-sims):
                    c = cf.pages[int(j)].page_number
                    if c not in (a, b):
                        wrong.setdefault(c, "most similar")
                        break
            for _ in range(sum(k == "random" for k in negatives)):
                others = [p for p in order if p not in (a, b) and p not in wrong]
                if others:
                    wrong.setdefault(rng.choice(others), "random")
            rows += [(texts[a][0], texts[c][1], 0, chart.name, wrong[c], a, c) for c in sorted(wrong)]
    return rows


def _held_out_auc(model, tokenizer, rows, max_length: int, batch: int = 64) -> Optional[tuple[float, dict, float]]:
    """AUC of the model on held-out pairs: against the kinds of wrong page it learns from
    (TRAIN_NEGATIVES -- the one the best epoch is chosen by), by kind, and against all kinds."""
    if not rows or len({r[2] for r in rows}) < 2:
        return None
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            enc = tokenizer([r[0] for r in chunk], [r[1] for r in chunk], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            probs += torch.sigmoid(model(**enc).logits.reshape(-1)).tolist()
    model.train()

    def auc(keep) -> float:
        sel = [(r[2], p) for r, p in zip(rows, probs) if r[2] or keep(r[4])]
        return round(float(roc_auc_score([y for y, _ in sel], [p for _, p in sel])), 4)

    by_kind = {kind: auc(lambda k, kind=kind: k == kind) for kind in NEGATIVE_KINDS if any(r[4] == kind for r in rows)}
    return auc(lambda k: k in TRAIN_NEGATIVES), by_kind, auc(lambda k: True)


def train_seam(base_path: Path, out_path: Path, examples: Iterable, epochs: int = 3, lr: float = 5e-5,
               batch: int = 32, max_length: int = 256, seed: int = 0, held_out: Iterable = (),
               resume: bool = False) -> dict:
    """Fine-tune on ``examples`` ((chart, truth) pairs). ``held_out``: other documents, scored after
    every epoch (AUC, also by kind of wrong page); the best epoch is the one saved. ``resume``: go on
    from the model already in ``out_path`` (same documents) for ``epochs`` more epochs."""
    import time

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rng = random.Random(seed)
    torch.manual_seed(seed)
    examples = list(examples)
    names = sorted({c.name for c, _ in examples})
    held = list(held_out)
    held_names = sorted({c.name for c, _ in held})
    before = {}
    if resume:
        before = json.loads((Path(out_path) / META).read_text(encoding="utf-8"))
        saved = set(before.get("trained_on") or [])
        common = saved & set(names)
        if len(common) < 0.9 * max(len(names), 1):
            raise ValueError(
                f"resume: the documents differ from the ones the saved seam learned from (saved: {len(saved)}, "
                f"now: {len(names)}, in common: {len(common)}; is SEQ_CONFIG set to the training "
                "shuffle set's config_shuffled.py?)")
        if saved != set(names):
            logger.info("seam resume: saved %d documents, now %d, in common %d -- going on; every document "
                        "it has learned from stays excluded from v8 / v9's fit", len(saved), len(names), len(common))
        rng = random.Random(seed + int(before.get("epochs") or 0))
    train_rows = training_pairs(examples, rng)
    # the held-out pairs include the "most similar" kind too, to see the seam against every kind
    test_rows = training_pairs(held, random.Random(seed), negatives=NEGATIVE_KINDS) if held else []

    start = Path(out_path) if resume else Path(base_path)
    tokenizer = AutoTokenizer.from_pretrained(str(start))
    model = AutoModelForSequenceClassification.from_pretrained(str(start), num_labels=1)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rescored = None
    if resume and test_rows:  # the saved epoch's score, measured the way this run measures
        rescored = _held_out_auc(model, tokenizer, test_rows, max_length)
        logger.info("seam: saved epoch %s re-scored: held-out AUC %s", before.get("epochs"),
                    rescored[0] if rescored else None)
    model.train()

    def save(meta_extra: dict) -> None:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        meta.update(meta_extra)
        (Path(out_path) / META).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    done_before = int(before.get("epochs") or 0)
    # resume goes on from the saved (best) epoch; later epochs that were not kept are dropped
    curve = [c for c in before.get("epoch_curve") or [] if c["epoch"] <= done_before]
    best = (rescored[0] if rescored else before.get("held_out_auc")) if resume else None
    # every document the seam has ever learned from (v8 / v9 leave all of them out of their fit)
    meta = {"base_model": before.get("base_model", str(base_path)),
            "trained_on": sorted(set(names) | set(before.get("trained_on") or [])),
            "held_out_documents": held_names, "pairs": len(train_rows), "held_out_pairs": len(test_rows),
            "negatives": list(TRAIN_NEGATIVES), "lr": lr, "max_length": max_length,
            "held_out_auc": best, "held_out_auc_by_kind": before.get("held_out_auc_by_kind"),
            "epochs": done_before, "epoch_curve": curve}
    batches = (len(train_rows) + batch - 1) // batch
    every = max(1, batches // 20)  # about 20 progress lines per epoch
    logger.info("seam: %d training pairs from %d documents; %d held-out pairs from %d documents; "
                "%d batches per epoch, %d epochs, input length %d, lr %g",
                len(train_rows), len(names), len(test_rows), len(held_names), batches, epochs, max_length, lr)
    started = time.time()
    done = 0
    for epoch in range(done_before + 1, done_before + epochs + 1):
        rng.shuffle(train_rows)
        total = 0.0
        for k, i in enumerate(range(0, len(train_rows), batch), start=1):
            chunk = train_rows[i:i + batch]
            enc = tokenizer([r[0] for r in chunk], [r[1] for r in chunk], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            labels = torch.tensor([float(r[2]) for r in chunk])
            loss = loss_fn(model(**enc).logits.reshape(-1), labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(chunk)
            done += 1
            if k % every == 0 or k == batches:
                elapsed = time.time() - started
                left = elapsed / done * (batches * epochs - done)
                logger.info("seam epoch %d/%d batch %d/%d  loss %.4f  elapsed %dm  about %dm left",
                            epoch, done_before + epochs, k, batches,
                            total / min(len(train_rows), i + len(chunk)), elapsed // 60, left // 60)
        epoch_loss = round(total / max(1, len(train_rows)), 4)
        got = _held_out_auc(model, tokenizer, test_rows, max_length) if test_rows else None
        auc, by_kind, auc_all = got if got else (None, None, None)
        curve.append({"epoch": epoch, "loss": epoch_loss, "held_out_auc": auc, "by_kind": by_kind,
                      "all_kinds": auc_all})
        logger.info("seam epoch %d: loss %.4f  held-out AUC %s (all kinds %s)  %s", epoch, epoch_loss, auc, auc_all,
                    "  ".join(f"vs {k} {v}" for k, v in (by_kind or {}).items()))
        if auc is None or best is None or auc > best:
            best = auc
            save({"held_out_auc": auc, "held_out_auc_by_kind": by_kind, "held_out_auc_all_kinds": auc_all,
                  "epochs": epoch, "epoch_curve": curve})
            logger.info("seam: epoch %d saved (best so far)", epoch)
        else:
            meta["epoch_curve"] = curve
            (Path(out_path) / META).write_text(json.dumps(meta, indent=1), encoding="utf-8")
            logger.info("seam: epoch %d not saved (epoch %d is better)", epoch, meta["epochs"])
    return {k: v for k, v in meta.items() if k not in ("trained_on", "held_out_documents")}
