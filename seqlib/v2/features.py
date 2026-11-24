"""v2 pair features: evidence that page B directly follows page A.

Built on v1's per-page reading (``seqlib.v1.runner.continuity_inputs`` -> the continuity
engine's PageInput: positioned lines, printed page N of M, DOS, page type, signature, title),
so every signal v1 uses is available to the model -- but here a learned model weighs them
instead of hand-set points, and they are computed for *any* pair of pages, not just scan
neighbours.

A virtual START page closes the order into a loop: (START, B) means "B is the first page",
(A, START) means "A is the last page".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from seqlib.io.chart import Chart
from seqlib.v1.continuity.engine import (
    PageInput, body, body_end, continued_section, foot, head, load_rules, pagination, visit_heading,
)
from seqlib.v1.continuity.signals import (
    cut_mid_sentence, head_block, normalized_line, says_continued, starts_mid_thought, tail_block, visit_numbers,
)
from seqlib.v1.runner import EXCLUDE, continuity_inputs, documents

START = -1  # the virtual page

FEATURES = (
    "a_is_start", "b_is_start",
    "scan_delta", "abs_scan_delta", "is_scan_next", "is_scan_prev",
    "both_printed", "same_total", "printed_step", "printed_follows", "a_last_printed", "b_first_printed",
    "a_printed_n", "b_printed_n",
    "both_dos", "same_dos", "dos_days",
    "same_family", "same_names",
    "head_jaccard", "foot_jaccard", "shared_furniture",
    "tail_head_char", "tail_head_word", "page_char", "tail_head_char_rank", "tail_head_char_rank_rev",
    "a_cut_mid", "b_starts_lower", "a_says_continued", "b_continues_a_section",
    "a_signed_tail", "a_signed", "b_title", "b_visit_heading",
    "same_visit_id",
    "same_doc_v1", "doc_delta_v1",
    "a_fill", "a_words", "b_words",
    "a_rel_pos", "b_rel_pos", "n_pages",
    "emb_tail_head",  # always empty now (the sentence embedder was removed); kept so trained models load
)
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


@dataclass
class PageInfo:
    page_number: int
    rank: int  # scan position among the chart's Main pages
    inp: PageInput
    printed: Optional[tuple[int, int]]
    head: set[str]
    foot: set[str]
    tail_text: str
    head_text: str
    words: int
    fill: float
    visit_ids: frozenset[str]
    heading: bool
    cut_mid: bool
    starts_lower: bool
    says_cont: bool
    doc_v1: Optional[int]


@dataclass
class ChartFeatures:
    pages: list[PageInfo]
    index: dict[int, int]  # page_number -> row in the matrices
    char_sim: np.ndarray  # tail(a) . head(b)
    word_sim: np.ndarray
    page_sim: np.ndarray  # whole page vs whole page
    emb_sim: Optional[np.ndarray] = None
    notes: list[str] = field(default_factory=list)


def _days(a: str, b: str) -> Optional[float]:
    ma, mb = _DATE.match(a or ""), _DATE.match(b or "")
    if not (ma and mb):
        return None
    from datetime import date

    try:
        return float((date(*map(int, mb.groups())) - date(*map(int, ma.groups()))).days)
    except ValueError:
        return None


def _cosines(left: Sequence[str], right: Sequence[str], analyzer: str, ngram: tuple[int, int]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(left)
    if n == 0:
        return np.zeros((0, 0))
    docs = list(left) + list(right)
    if not any(d.strip() for d in docs):
        return np.zeros((n, len(right)))
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, min_df=1, lowercase=True)
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return np.zeros((n, len(right)))
    return (matrix[:n] @ matrix[n:].T).toarray()


def order_free(chart: Chart) -> Chart:
    """A copy of ``chart`` whose pipeline rows lack every value computed from the page order
    (seqlib.eval.shuffle.ORDER_DEPENDENT: document_seq, seq, the document's agreed type / DOS /
    confidences ...). v2 reads only per-page values, so it sees the same kind of input whatever
    order the pages came in -- on real data, on a shuffle test, and in production."""
    import copy

    from seqlib.eval.shuffle import strip_order_dependent

    pages = []
    for page in chart.pages:
        clone = copy.copy(page)
        clone.rows = dict(page.rows)
        if page.pipeline:
            clone.rows["pipeline"] = strip_order_dependent(page.pipeline)
        pages.append(clone)
    out = copy.copy(chart)
    out.pages = pages
    return out


def page_infos(chart: Chart, pages: Optional[Sequence[int]] = None) -> list[PageInfo]:
    """Per-page facts for the pages to order (default: the pipeline's Main pages)."""
    chart = order_free(chart)
    rules = load_rules()
    inputs = continuity_inputs(chart)
    try:
        # documents as v1's continuity groups the pages in the order given (document_seq is
        # order-dependent, so it is not read)
        docs = documents(chart, "recompute")
    except Exception:  # noqa: BLE001 - v1 grouping is one feature among many
        docs = {}
    by_number = {i.page_number: i for i in inputs}
    if pages is None:
        pages = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    wanted = sorted(set(pages))
    out: list[PageInfo] = []
    n = rules.semantic.tail_nth_word
    for rank, number in enumerate(wanted):
        inp = by_number.get(number) or PageInput(page_id=number, page_number=number)
        if inp.kind != "main":
            # a page the pipeline set aside but the truth (training) keeps: read it as Main
            page = chart.by_number()[number]
            from seqlib.v1.continuity.engine import text_lines

            inp.kind, inp.text = "main", page.text
            inp.lines = inp.lines or text_lines(page.text)
        body_lines = body(inp, rules)
        body_text = [l.text for l in body_lines]
        end = body_end(inp, rules)
        out.append(PageInfo(
            page_number=number,
            rank=rank,
            inp=inp,
            printed=pagination(inp, rules),
            head={normalized_line(t) for t in head(inp, rules) if len(t) >= 12},
            foot={normalized_line(t) for t in foot(inp, rules) if len(t) >= 12},
            tail_text=tail_block(body_text, n * 2),
            head_text=head_block(body_text, n * 2),
            words=len(inp.text.split()),
            fill=end if end is not None else float("nan"),
            visit_ids=visit_numbers(inp.text),
            heading=bool(visit_heading(inp, rules)),
            cut_mid=bool(body_text) and cut_mid_sentence(body_text[-1]),
            starts_lower=bool(body_lines) and starts_mid_thought(body_lines[0].raw),
            says_cont=says_continued((body_text or [l.text for l in inp.lines])[-2:]),
            doc_v1=docs.get(number),
        ))
    return out


def chart_features(chart: Chart, pages: Optional[Sequence[int]] = None) -> ChartFeatures:
    infos = page_infos(chart, pages)
    tails = [p.tail_text for p in infos]
    heads = [p.head_text for p in infos]
    whole = [p.inp.text for p in infos]
    cf = ChartFeatures(
        pages=infos,
        index={p.page_number: i for i, p in enumerate(infos)},
        char_sim=_cosines(tails, heads, "char_wb", (3, 5)),
        word_sim=_cosines(tails, heads, "word", (1, 2)),
        page_sim=_cosines(whole, whole, "word", (1, 1)),
    )
    return cf


def _rank_rows(sim: np.ndarray) -> np.ndarray:
    """For each (a, b): b's rank among a's candidates by similarity (0 = most similar)."""
    order = np.argsort(-sim, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(sim.shape[0])[:, None]
    ranks[rows, order] = np.arange(sim.shape[1])[None, :]
    return ranks


def candidates(cf: ChartFeatures, window: int, top_k: int) -> list[tuple[int, int]]:
    """Pairs (a, b) worth scoring, as page numbers (START = -1): scan neighbours within
    ``window``, the ``top_k`` most similar tail/head matches, printed-number successors, and
    START with every page."""
    n = len(cf.pages)
    pairs: set[tuple[int, int]] = set()
    numbers = [p.page_number for p in cf.pages]
    for i, a in enumerate(cf.pages):
        pairs.add((START, a.page_number))
        pairs.add((a.page_number, START))
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j != i:
                pairs.add((a.page_number, numbers[j]))
        if n > 1 and top_k:
            sims = cf.char_sim[i].copy()
            sims[i] = -1
            for j in np.argsort(-sims)[:top_k]:
                pairs.add((a.page_number, numbers[int(j)]))
        if a.printed:
            for j, b in enumerate(cf.pages):
                if j != i and b.printed and b.printed[1] == a.printed[1] and b.printed[0] == a.printed[0] + 1:
                    pairs.add((a.page_number, b.page_number))
    return sorted(pairs)


def pair_matrix(cf: ChartFeatures, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    n = len(cf.pages)
    char_rank = _rank_rows(cf.char_sim) if n else None
    char_rank_rev = _rank_rows(cf.char_sim.T).T if n else None
    rows = np.full((len(pairs), len(FEATURES)), np.nan, dtype=np.float32)
    col = {name: k for k, name in enumerate(FEATURES)}
    for r, (an, bn) in enumerate(pairs):
        x = rows[r]
        x[col["n_pages"]] = n
        a = cf.pages[cf.index[an]] if an != START else None
        b = cf.pages[cf.index[bn]] if bn != START else None
        x[col["a_is_start"]] = a is None
        x[col["b_is_start"]] = b is None
        ra = a.rank if a else -1
        rb = b.rank if b else n
        x[col["scan_delta"]] = rb - ra
        x[col["abs_scan_delta"]] = abs(rb - ra)
        x[col["is_scan_next"]] = rb - ra == 1
        x[col["is_scan_prev"]] = rb - ra == -1
        if a:
            x[col["a_rel_pos"]] = a.rank / max(1, n - 1)
            x[col["a_words"]] = a.words
            x[col["a_fill"]] = a.fill
            x[col["a_cut_mid"]] = a.cut_mid
            x[col["a_says_continued"]] = a.says_cont
            x[col["a_signed_tail"]] = a.inp.signed_in_tail
            x[col["a_signed"]] = a.inp.signed
            if a.printed:
                x[col["a_printed_n"]] = a.printed[0]
                x[col["a_last_printed"]] = a.printed[0] == a.printed[1]
        if b:
            x[col["b_rel_pos"]] = b.rank / max(1, n - 1)
            x[col["b_words"]] = b.words
            x[col["b_starts_lower"]] = b.starts_lower
            x[col["b_title"]] = b.inp.title_hit
            x[col["b_visit_heading"]] = b.heading
            if b.printed:
                x[col["b_printed_n"]] = b.printed[0]
                x[col["b_first_printed"]] = b.printed[0] == 1
        if not (a and b):
            continue
        ia, ib = cf.index[an], cf.index[bn]
        if a.printed and b.printed:
            x[col["both_printed"]] = 1
            same = a.printed[1] == b.printed[1]
            x[col["same_total"]] = same
            if same:
                step = b.printed[0] - a.printed[0]
                x[col["printed_step"]] = max(-20, min(20, step))
                x[col["printed_follows"]] = step == 1
        else:
            x[col["both_printed"]] = 0
        if a.inp.dos_from and b.inp.dos_from:
            x[col["both_dos"]] = 1
            x[col["same_dos"]] = a.inp.dos == b.inp.dos
            days = _days(a.inp.dos_from, b.inp.dos_from)
            if days is not None:
                x[col["dos_days"]] = max(-3650, min(3650, days))
        else:
            x[col["both_dos"]] = 0
        if a.inp.page_type and b.inp.page_type:
            x[col["same_family"]] = a.inp.page_type == b.inp.page_type
        if a.inp.names and b.inp.names:
            x[col["same_names"]] = a.inp.names == b.inp.names
        x[col["head_jaccard"]] = _jaccard(a.head, b.head)
        x[col["foot_jaccard"]] = _jaccard(a.foot, b.foot)
        x[col["shared_furniture"]] = len((a.head | a.foot) & (b.head | b.foot))
        x[col["tail_head_char"]] = cf.char_sim[ia, ib]
        x[col["tail_head_word"]] = cf.word_sim[ia, ib]
        x[col["page_char"]] = cf.page_sim[ia, ib]
        x[col["tail_head_char_rank"]] = min(50, int(char_rank[ia, ib]))
        x[col["tail_head_char_rank_rev"]] = min(50, int(char_rank_rev[ia, ib]))
        x[col["b_continues_a_section"]] = bool(continued_section(a.inp, b.inp, load_rules()))
        x[col["same_visit_id"]] = bool(a.visit_ids & b.visit_ids)
        if a.doc_v1 is not None and b.doc_v1 is not None:
            x[col["same_doc_v1"]] = a.doc_v1 == b.doc_v1
            x[col["doc_delta_v1"]] = max(-20, min(20, b.doc_v1 - a.doc_v1))
        if cf.emb_sim is not None:
            x[col["emb_tail_head"]] = cf.emb_sim[ia, ib]

    return rows


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return math.nan
    return len(a & b) / len(a | b)
