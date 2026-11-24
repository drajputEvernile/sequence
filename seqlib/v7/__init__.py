"""The gated versions (v8, v9; seqlib/versions.py): before ordering, a gate estimates whether the
document is out of order at all, and picks the move margin that suits -- a low one for documents
it flags, a high one for the rest.

What v3 to v6 showed: the evidence repairs more with a low margin but then also breaks documents
that were in order; a high margin protects them and repairs less. The gate reads document-level
signs on the scan order: printed page numbers / fax counters the scan order breaks, scan links
the evidence finds weak, how much cheaper the best order is than the scan order, dates jumping
back, v1's departures, printed numbers not stepping by one.

Everything is tuned on out-of-fold scores of the training documents (cross-fitting,
seqlib/v3/tuning.py), so the gate, its threshold and both margins are chosen on documents the
evidence model did not learn from.
"""
