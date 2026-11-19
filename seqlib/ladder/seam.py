"""The seam cross-encoder (design doc 8.3): P(page b's opening continues page a's ending).

A small pre-trained language model (for example all-MiniLM-L6-v2, Apache-2.0) read as a
cross-encoder: input "[CLS] a's last lines [SEP] b's first lines", one output. Fine-tuned on the
training documents' true neighbour pairs against wrong ones (the scan neighbour when it is wrong,
the most similar-looking page, a random page of the same document).

It is trained on one half of the training documents (the seam half); v6's evidence model then
trains on the other half, so the seam's scores on its own training pages cannot inflate v6.

Needs ``torch`` and ``transformers`` (requirements-seam.txt) and a local copy of the base model
(config.SEAM_BASE_MODEL_PATH); nothing is downloaded. Trains on a GPU when there is one.
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
    def __init__(self, path: Path, device: str = "cpu", max_length: Optional[int] = None, batch: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device).eval()
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
                                     max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.reshape(-1)
                out.append(self.torch.sigmoid(logits).cpu().numpy())
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


def training_pairs(examples: Iterable, rng: random.Random) -> list[tuple[str, str, int, str]]:
    """(a's ending, b's opening, label, document) from (chart, truth) examples."""
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
            rows.append((texts[a][0], texts[b][1], 1, chart.name))
            wrong = set()
            if nxt_scan.get(a) not in (None, b):
                wrong.add(nxt_scan[a])
            sims = cf.char_sim[idx[a]].copy()
            for j in np.argsort(-sims):
                c = cf.pages[int(j)].page_number
                if c not in (a, b):
                    wrong.add(c)
                    break
            others = [p for p in order if p not in (a, b) and p not in wrong]
            if others:
                wrong.add(rng.choice(others))
            rows += [(texts[a][0], texts[c][1], 0, chart.name) for c in sorted(wrong)]
    return rows


def train_seam(base_path: Path, out_path: Path, examples: Iterable, epochs: int = 3, lr: float = 2e-5,
               batch: int = 32, max_length: int = 256, device: str = "cpu", seed: int = 0,
               held_share: float = 0.1) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rng = random.Random(seed)
    torch.manual_seed(seed)
    examples = list(examples)
    names = sorted({c.name for c, _ in examples})
    held = set(rng.sample(names, max(1, int(len(names) * held_share)))) if len(names) >= 5 else set()
    rows = training_pairs(examples, rng)
    train_rows = [r for r in rows if r[3] not in held]
    test_rows = [r for r in rows if r[3] in held]
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(base_path), num_labels=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    import time

    batches = (len(train_rows) + batch - 1) // batch
    every = max(1, batches // 20)  # about 20 progress lines per epoch
    logger.info("seam: %d training pairs (%d held out), %d batches per epoch, %d epochs, input length %d, %s",
                len(train_rows), len(test_rows), batches, epochs, max_length, device)
    started = time.time()
    done = 0
    for epoch in range(epochs):
        rng.shuffle(train_rows)
        total = 0.0
        for k, i in enumerate(range(0, len(train_rows), batch), start=1):
            chunk = train_rows[i:i + batch]
            enc = tokenizer([r[0] for r in chunk], [r[1] for r in chunk], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([float(r[2]) for r in chunk], device=device)
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
                            epoch + 1, epochs, k, batches, total / min(len(train_rows), i + len(chunk)),
                            elapsed // 60, left // 60)
        logger.info("seam epoch %d: loss %.4f", epoch + 1, total / max(1, len(train_rows)))
    model.eval()
    Path(out_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    auc = None
    if test_rows and len({r[2] for r in test_rows}) == 2:
        from sklearn.metrics import roc_auc_score

        scorer = SeamScorer(out_path, device, max_length)
        probs = scorer.score([r[0] for r in test_rows], [r[1] for r in test_rows])
        auc = round(float(roc_auc_score([r[2] for r in test_rows], probs)), 4)
    meta = {"base_model": str(base_path), "trained_on": names, "held_out_documents": sorted(held),
            "pairs": len(train_rows), "held_out_pairs": len(test_rows), "held_out_auc": auc,
            "epochs": epochs, "lr": lr, "max_length": max_length, "device": device}
    (Path(out_path) / META).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {k: v for k, v in meta.items() if k not in ("trained_on", "held_out_documents")}
