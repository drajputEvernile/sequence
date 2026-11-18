"""Ported unchanged from model-repo tests/test_continuity.py (see seqlib/v1/SOURCE_COMMIT), imports aside.
Tests that need the pipeline's database, staging files or schema are left out: TestSchema, TestStage, TestStaging, test_imaging_final_gives_junk_pages_no_document_values."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest


from seqlib.v1.continuity.engine import (  # noqa: E402
    DECIDED_BY,
    Line,
    PageInput,
    assign,
    load_rules,
    pagination,
    text_lines,
    trust_points,
)
from seqlib.v1.continuity.layout import lines_from_final1, lines_from_final2  # noqa: E402
from seqlib.v1.continuity.signals import fill_value, normalized_line  # noqa: E402

BANNER = "patient: jane doe dob 01/02/1950 mrn 123456"
FOOTER = "community clinic 100 main street springfield"
RULES = load_rules()


def page(n, body="", *, head=BANNER, foot=FOOTER, ends=0.80, **kw):
    """A located page: head line at the top, body ending ``ends`` down the page, foot at
    the bottom. ``ends`` 0.80 is where the page-fill curve says nothing."""
    lines = []
    if head:
        lines.append(Line(head, 0.03, 0.05))
    for i, text in enumerate((body or f"k{n}v{n}j").splitlines()):
        lines.append(Line(text.lower(), 0.30 + i * 0.02, 0.31 + i * 0.02))
    # A closing line of its own, so pages share no wording unless a test gives them some.
    lines.append(Line(f"q{n}x{n}z", ends - 0.01, ends))
    if foot:
        lines.append(Line(foot, 0.95, 0.97))
    return PageInput(
        page_id=n, page_name=f"{n}.jpg", page_number=n,
        text="\n".join(l.text for l in lines), lines=lines, **kw,
    )


def bare(n, body="", **kw):
    return page(n, body, head="", foot="", **kw)


def junk(n, subtype="Record Request/Transmittal", confidence=0.96, **kw):
    return bare(n, f"records request {n}", kind="junk", junk_subtype=subtype, junk_confidence=confidence, **kw)


def gap(row):
    return row["signals"]["before"]


class TestDecisive:
    def test_page_numbers_running_on_join(self):
        rows = assign([bare(1, printed=(1, 2)), bare(2, printed=(2, 2))])
        assert rows[1]["relation"] == "continue" and rows[1]["decided_by"] == "decisive"
        assert gap(rows[1])["pair"]["pagination_follows"] == -5
        assert rows[1]["confidence_level"] == "high" and rows[1]["confidence"] >= 0.9

    def test_a_different_labelled_date_breaks(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5), bare(2, dos_from="2024-02-01", dos_trust=5)])
        assert (rows[1]["relation"], rows[1]["decided_by"]) == ("new_document", "decisive")

    def test_an_edge_keyword_date_is_strong_not_decisive(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5), bare(2, dos_from="2024-02-01", dos_trust=3)])
        assert gap(rows[1])["pair"]["different_dos"] == 3 and rows[1]["decided_by"] == "sum"

    def test_the_same_date_joins(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5), bare(2, dos_from="2024-01-01", dos_trust=5)])
        assert rows[1]["relation"] == "continue" and gap(rows[1])["pair"]["same_dos"] == -5

    def test_a_date_beats_page_numbers_at_medium_confidence(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5, printed=(1, 2)),
                       bare(2, dos_from="2024-02-01", dos_trust=5, printed=(2, 2))])
        assert rows[1]["relation"] == "new_document" and rows[1]["decided_by"] == "decisive_conflict"
        # Two decisive signals disagree: the conflict penalty puts it under the review line.
        assert rows[1]["confidence_level"] == "low" and rows[1]["review_required"]

    def test_dos_trust_by_how_the_date_was_found(self):
        assert [trust_points(t, RULES) for t in ("service", "admit", "weak", "keyless", "collected", None)] == [5, 5, 3, 2, 3, 3]


class TestEvidence:
    def test_a_signed_last_page_then_a_title_breaks(self):
        rows = assign([bare(1, signed=True, signed_in_tail=True, printed=(2, 2)), bare(2, title_hit=True)])
        before = gap(rows[1])
        assert rows[1]["relation"] == "new_document"
        assert before["ending"]["signed_in_tail"] == 3 and before["ending"]["signed_anywhere"] == 2
        assert before["ending"]["last_numbered_page"] == 3 and before["opening"]["has_title"] == 3

    def test_the_page_fill_curve(self):
        cfg = RULES.fill
        assert fill_value(0.95, cfg) == -1 and fill_value(0.60, cfg) == 1 and fill_value(0.20, cfg) == 0
        assert -1 < fill_value(0.86, cfg) < 0 and 0 < fill_value(0.72, cfg) < 1 and 0 < fill_value(0.42, cfg) < 1

    def test_a_tie_joins_and_is_flagged(self):
        rows = assign([bare(1), bare(2)])
        assert rows[1]["relation"] == "continue" and rows[1]["decided_by"] == "tie"
        assert rows[1]["review_required"]

    def test_header_lines_match_once_numbers_are_placeholders(self):
        assert normalized_line("Smith, John | MRN 004812 | Printed 03/28/24 | Page 3 of 7") == (
            "smith, john | mrn # | printed <date> | page # of #")
        rows = assign([page(1, head="smith john mrn 004812 printed 03/28/24 visit summary"),
                       page(2, head="smith john mrn 004812 printed 03/29/24 visit summary")])
        assert gap(rows[1])["pair"]["same_header_line"] == -2

    def test_shared_furniture_is_not_wording(self):
        rows = assign([page(1, "aaaa bbbb cccc dddd"), page(2, "xxxx yyyy wwww vvvv")])
        assert "wording_char" not in gap(rows[1])["pair"]

    def test_the_next_page_start_is_recorded_not_scored(self):
        rows = assign([page(1), page(2)])
        assert "text_start" in gap(rows[1])["recorded"] and "text_start" not in gap(rows[1])["opening"]


class TestTransparentPages:
    def test_a_blank_page_is_scored_across_with_one_point(self):
        rows = assign([bare(1, printed=(1, 2)), bare(2, kind="blank"), bare(3, printed=(2, 2))])
        assert gap(rows[2])["pair"]["blank_between"] == 1
        assert rows[1]["label"] == "Blank (in Doc 1)" and rows[1]["break_before"] is None

    def test_a_blank_page_between_documents(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5), bare(2, kind="blank"),
                       bare(3, dos_from="2024-02-01", dos_trust=5)])
        assert rows[1]["label"] == "Blank" and rows[1]["segment_kind"] is None

    def test_a_single_junk_page_scores_nothing_and_stays_inside(self):
        rows = assign([bare(1, printed=(1, 2)), junk(2), bare(3, printed=(2, 2))])
        assert "junk_run" not in gap(rows[2])["pair"]
        assert rows[1]["segment_kind"] == "main" and rows[1]["document_seq"] == 1
        assert rows[1]["label"] == "Continue (Doc 1)"

    def test_a_run_of_two_junk_pages_is_its_own_segment(self):
        rows = assign([bare(1, signed=True), junk(2), junk(3), bare(4, title_hit=True)])
        # Documents and junk groups share the numbering (document_seq).
        assert [r["label"] for r in rows[1:3]] == ["New (Junk 2)", "Last (Junk 2)"]
        assert [r["document_seq"] for r in rows] == [1, 2, 2, 3]
        assert gap(rows[1])["pair"]["junk_run"] == pytest.approx(3 * 0.96)
        assert rows[3]["segment_kind"] == "main"

    def test_a_run_of_three_junk_pages_is_decisive(self):
        rows = assign([bare(1), junk(2), junk(3), junk(4), bare(5)])
        assert rows[1]["decided_by"] == "decisive" and rows[4]["decided_by"] == "decisive"
        assert {r["segment_kind"] for r in rows[1:4]} == {"junk"}

    def test_a_duplicate_takes_its_originals_document(self):
        rows = assign([bare(1, printed=(1, 2)), bare(2, printed=(2, 2)),
                       bare(3, kind="duplicate", duplicate_of=2)])
        assert rows[2]["segment_seq"] == rows[1]["segment_seq"] == 1


class TestTags:
    def test_new_continue_last_standalone(self):
        rows = assign([bare(1, printed=(1, 3)), bare(2, printed=(2, 3)), bare(3, printed=(3, 3)),
                       bare(4, dos_from="2024-02-01", dos_trust=5, title_hit=True, signed=True, signed_in_tail=True),
                       bare(5, dos_from="2024-03-01", dos_trust=5)])
        assert [r["tag"] for r in rows] == ["new", "continue", "last", "standalone", "standalone"]
        assert [r["label"] for r in rows[:3]] == ["New (Doc 1)", "Continue (Doc 1)", "Last (Doc 1)"]
        assert rows[3]["break_before"] == rows[2]["break_after"]

    def test_a_low_confidence_side_adds_a_question_mark(self):
        rows = assign([bare(1), bare(2)])
        assert rows[0]["label"] == "New? (Doc 1)" and rows[1]["label"] == "Last? (Doc 1)"


class TestFinalPass:
    def test_a_document_with_two_labelled_dates_is_split(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=3, printed=(1, 3)),
                       bare(2, dos_from="2024-02-01", dos_trust=5, printed=(2, 3)),
                       bare(3, dos_from="2024-03-01", dos_trust=5, printed=(3, 3))])
        assert rows[2]["relation"] == "new_document" and rows[2]["decided_by"] == "final_pass"
        assert "split" in gap(rows[2])["final_pass"]

    def test_a_weak_break_between_same_dated_documents_is_merged(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=2),
                       bare(2, dos_from="2024-01-01", dos_trust=2, title_hit=True)])
        assert gap(rows[1])["score"] == 1
        assert rows[1]["relation"] == "continue" and gap(rows[1])["final_pass"] == "merged: same_date"

    def test_same_type_untitled_neighbours_are_merged(self):
        rows = assign([bare(1, names=("Lab",), page_type="Lab", ends=0.6),
                       bare(2, names=("Lab",), page_type="Lab")])
        assert gap(rows[1])["score"] == 1
        assert rows[1]["relation"] == "continue" and "same_type_untitled" in gap(rows[1])["final_pass"]

    def test_a_decisive_break_is_never_merged(self):
        rows = assign([bare(1, dos_from="2024-01-01", dos_trust=5), bare(2, dos_from="2024-02-01", dos_trust=5)])
        assert rows[1]["relation"] == "new_document" and not gap(rows[1])["final_pass"]


class TestProgressNoteRunsOn:
    """A document opening on a Progress Note keeps going until a strong date or a signature."""

    def test_it_absorbs_the_pages_after_it(self):
        rows = assign([bare(1, page_type="Progress Note", dos_from="2024-01-01", dos_trust=5, ends=0.6),
                       bare(2, page_type="Lab Report", page_confidence=0.9, title_hit=True),
                       bare(3, page_type="Questionnaire", page_confidence=0.9, title_hit=True)])
        assert [r["document_seq"] for r in rows] == [1, 1, 1]
        assert gap(rows[1])["final_pass"] == "merged: progress_note_runs_on"

    def test_a_signature_ends_it(self):
        rows = assign([bare(1, page_type="Progress Note", signed=True, signed_in_tail=True, ends=0.6),
                       bare(2, page_type="Lab Report", page_confidence=0.9, title_hit=True)])
        assert [r["document_seq"] for r in rows] == [1, 2]

    def test_a_new_labelled_date_ends_it(self):
        rows = assign([bare(1, page_type="Progress Note", dos_from="2024-01-01", dos_trust=5),
                       bare(2, page_type="Lab Report", page_confidence=0.9, title_hit=True, ends=0.6),
                       bare(3, page_type="Progress Note", dos_from="2024-02-01", dos_trust=5)])
        assert [r["document_seq"] for r in rows] == [1, 1, 2]

    def test_a_junk_group_ends_it(self):
        rows = assign([bare(1, page_type="Progress Note"), junk(2), junk(3), junk(4), bare(5)])
        assert rows[4]["document_seq"] != rows[0]["document_seq"]


class TestConflicts:
    def _pages(self):
        visits = frozenset({"fin 100234"})
        return [page(1, signed=True, signed_in_tail=True, ends=0.6, visit_ids=visits),
                page(2, visit_ids=visits)]

    # The scored conflict, with the signature break (2026-11-01) switched off: it would
    # decide the gap first.
    NO_SIGNATURE_BREAK = dataclasses.replace(RULES, signature_breaks=False)

    def test_pattern_a_is_recorded_and_flagged_not_applied(self):
        rows = assign(self._pages(), self.NO_SIGNATURE_BREAK)
        before = gap(rows[1])
        assert before["recorded"]["conflict"] == "A" and rows[1]["review_required"]
        assert before["decided_by"] == "sum"

    def test_pattern_a_breaks_when_switched_on(self):
        rules = dataclasses.replace(self.NO_SIGNATURE_BREAK, conflict_apply=True)
        rows = assign(self._pages(), rules)
        assert rows[1]["decided_by"] == "conflict_a" and rows[1]["relation"] == "new_document"


class TestPagination:
    def test_only_the_header_and_footer_are_searched(self):
        body = page(1, "see page 2 of 4 of the attached report", head="x", foot="y")
        assert pagination(body, RULES) is None
        footer = page(1, "body", foot="printed 10/30/2025 pg 8 of 17")
        assert pagination(footer, RULES) == (8, 17)

    def test_page_1_confirms_the_start(self):
        rows = assign([bare(1, printed=(1, 2)), bare(2, printed=(2, 2))])
        assert rows[0]["start_confirmed"] is True and rows[1]["page_no"] == 2
        assert rows[1]["link_strength"] == "strong"


class TestFinalValues:
    def test_continue_pages_take_the_first_pages_dos_and_type(self):
        rows = assign([page(1, dos_from="2024-03-14", dos_trust=5, page_type="Progress Note"),
                       page(2, page_type="Lab Report", page_confidence=0.5), page(3)])
        assert [r["final_dos_from"] for r in rows] == ["2024-03-14"] * 3
        assert [r["final_page_type"] for r in rows] == ["Progress Note"] * 3

    def test_a_stand_alone_non_visit_document_takes_the_default_dos(self):
        rows = assign([page(1, dos_from="2024-03-14", dos_trust=5, page_type="Patient Demographics"),
                       page(2, page_type="Patient Demographics", page_confidence=0.9)])
        assert [r["final_dos_from"] for r in rows] == ["2222-02-22"] * 2
        assert "is not a visit" in rows[0]["evidence"]

    def test_a_non_visit_page_inside_a_visit_takes_the_visit_dos(self):
        rows = assign([page(1, dos_from="2024-03-14", dos_trust=5, page_type="Progress Note"),
                       page(2, page_type="Problem List", page_confidence=0.6)])
        assert [r["final_dos_from"] for r in rows] == ["2024-03-14"] * 2

    def test_the_sub_type_decides_too(self):
        from seqlib.v1.continuity.engine import document_finals

        finals = document_finals([{"page_type": "Clinical Summary", "page_subtype": "Problem List",
                                   "dos_from": "2024-01-01"}])
        assert finals["dos_from"] == "2222-02-22" and finals["dos_default"]

    def test_dos_is_the_first_dated_pages(self):
        from seqlib.v1.continuity.engine import document_finals

        finals = document_finals([
            {"page_type": "Progress Note", "dos_from": ""},
            {"page_type": "Progress Note", "dos_from": "2024-03-05", "dos_to": "2024-03-06"},
            {"page_type": "Progress Note", "dos_from": "2024-04-01"},
        ])
        assert (finals["dos_from"], finals["dos_to"]) == ("2024-03-05", "2024-03-06")

    def test_dos_is_the_first_pages_when_it_is_dated(self):
        from seqlib.v1.continuity.engine import document_finals

        finals = document_finals([
            {"page_type": "Progress Note", "dos_from": "2024-01-01"},
            {"page_type": "Progress Note", "dos_from": "2024-03-05"},
        ])
        assert (finals["dos_from"], finals["dos_to"]) == ("2024-01-01", "2024-01-01")

    def test_encounter_type_is_the_first_pages(self):
        from seqlib.v1.continuity.engine import document_finals

        finals = document_finals([
            {"page_type": "Progress Note", "dos_from": "2024-01-01", "encounter_type": "Inpatient"},
            {"page_type": "Progress Note", "dos_from": "", "encounter_type": "Outpatient (F2F)"},
        ])
        assert finals["encounter_type"] == "Inpatient"


class TestLayout:
    def test_final2_lines_are_placed_by_their_polygons(self):
        lines = lines_from_final2({"pagesMeta": [{"height": 1000.0, "lines": [
            {"content": "Footer  Line", "polygon": [0, 950, 10, 950, 10, 970, 0, 970]},
            {"content": "Banner", "polygon": [0, 20, 10, 20, 10, 40, 0, 40]},
        ]}]})
        assert [l.text for l in lines] == ["banner", "footer line"]
        assert lines[0].top == 0.02 and lines[1].bottom == 0.97

    def test_final1_words_are_grouped_into_lines_left_to_right(self):
        word = lambda text, x, y: {"content": text, "polygon": [x, y, x + 5, y, x + 5, y + 10, x, y + 10]}
        lines = lines_from_final1({"height": 1000.0, "words": [
            word("world", 50, 101), word("hello", 0, 100), word("next", 0, 300),
        ]})
        assert [l.text for l in lines] == ["hello world", "next"]

    def test_text_alone_falls_back_to_first_and_last_lines(self):
        p = PageInput(page_id=1, lines=text_lines("Page 3 of 9\nbody\nbody\nbody"))
        assert pagination(p, RULES) == (3, 9)


class TestEncounterSignals:
    """2026-10-23 (user): visit headings open an encounter; Demographics stands alone."""

    def test_a_visit_heading_at_the_top_pushes_toward_a_break(self):
        rows = assign([bare(1), bare(2, "Chief Complaint: cough for two weeks")])
        before = gap(rows[1])
        assert before["opening"]["visit_heading"] == 3
        assert before["evidence"]["visit_heading"].lower().startswith("chief complaint")

    def test_a_visit_heading_at_the_top_always_breaks(self):
        # Same labelled DOS (-5, decisive join) would otherwise keep them together.
        note = dict(dos_from="2024-03-14", dos_trust=5)
        rows = assign([bare(1, **note), bare(2, "HPI: cough for two weeks", **note)])
        assert gap(rows[1])["decision"] == "break"
        assert rows[0]["document_seq"] != rows[1]["document_seq"]

    def test_a_visit_heading_on_a_page_that_runs_on_by_number_still_joins(self):
        rows = assign([bare(1, printed=(1, 2)), bare(2, "Reason for Appointment: follow up", printed=(2, 2))])
        assert rows[0]["document_seq"] == rows[1]["document_seq"]

    def test_a_visual_page_joins_the_page_before_it(self):
        demo = dict(page_type="Demographics", names=("Patient Demographics", "Demographics"))
        rows = assign([bare(1, "Chief Complaint: cough"), bare(2, "", visual=True, **demo)])
        assert gap(rows[1])["evidence"]["visual_join"]
        assert rows[0]["document_seq"] == rows[1]["document_seq"]

    def test_a_visit_heading_after_a_left_column_on_the_same_line_counts(self):
        # Two-column page: OCR joins the left column's words and the heading into one line.
        rows = assign([bare(1), bare(2, "Current Medications Reason for Appointment")])
        assert gap(rows[1])["evidence"]["visit_heading"]
        # A one-word term ("cc") only at the start of a line.
        rows = assign([bare(1), bare(2, "copies sent to the clinic cc")])
        assert "visit_heading" not in gap(rows[1])["opening"]

    def test_a_visit_heading_low_on_the_page_does_not_count(self):
        body = "\n".join(["assessment stable"] * 6 + ["Reason for Visit: follow up"])
        rows = assign([bare(1), bare(2, body)])
        assert "visit_heading" not in gap(rows[1])["opening"]

    def test_demographics_never_joins_a_progress_note(self):
        # Page numbers running on (-5, decisive join) would join these two.
        demo = bare(1, page_type="Demographics", names=("Patient Demographics", "Demographics"),
                    printed=(1, 2), dos_from="2024-03-14", dos_trust=5)
        note = bare(2, page_type="Progress Note", names=("Progress Note",),
                    printed=(2, 2), dos_from="2024-03-14", dos_trust=5)
        rows = assign([demo, note])
        assert gap(rows[1])["decision"] == "break"
        assert gap(rows[1])["evidence"]["demographics_boundary"]
        assert rows[0]["document_seq"] != rows[1]["document_seq"]
        # Standing alone, the Demographics document takes the default DOS.
        assert rows[0]["final_dos_from"] == "2222-02-22"
        assert rows[1]["final_dos_from"] == "2024-03-14"

    def test_consecutive_demographics_pages_stay_together(self):
        demo = dict(page_type="Demographics", names=("Patient Demographics", "Demographics"))
        rows = assign([bare(1, printed=(1, 2), **demo), bare(2, printed=(2, 2), **demo)])
        assert rows[0]["document_seq"] == rows[1]["document_seq"]


class TestStartsMidThought:
    """2026-10-28: lines are lower-cased for matching, so the check ran on lower-case text
    and fired on nearly every gap with shared wording. It now reads the line as written."""

    def _gap(self, first_line_as_read):
        a = page(1, "the patient reports chest pain radiating to the left arm since monday evening")
        b = page(2, f"{first_line_as_read}\nchest pain radiating to the left arm eased on monday evening")
        b.lines = [dataclasses.replace(l, raw=first_line_as_read) if l.text == first_line_as_read.lower() else l
                   for l in b.lines]
        return gap(assign([a, b])[1])

    def test_a_capitalised_first_line_does_not_fire(self):
        g = self._gap("Assessment and plan reviewed with the patient today")
        assert "wording_char" in g["pair"]
        assert "starts_mid_thought" not in g["opening"]

    def test_a_lower_case_first_line_fires(self):
        g = self._gap("and was advised to continue the current medication")
        assert "wording_char" in g["pair"]
        assert "starts_mid_thought" in g["opening"]

    def test_text_lines_keep_the_case_as_read(self):
        line = text_lines("  Chest Pain  \n")[0]
        assert (line.text, line.raw) == ("chest pain", "Chest Pain")


class TestSignatureAndSummaryBreaks:
    """2026-11-01 (user): a signed page ends its document (a mid-page signature too, and
    whatever a sticky '... visit (continued)' header, the date or the page numbers say);
    a summary page stands apart from other families."""

    STICKY = "07/29/2025 - procedure visit in clinic (continued)"

    def test_a_signed_page_breaks_after_it_despite_sticky_header_date_and_page_numbers(self):
        pages = [
            page(1, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(1, 3)),
            page(2, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(2, 3),
                 signed=True),
            page(3, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Lab Report", printed=(3, 3)),
        ]
        rows = assign(pages)
        assert rows[1]["relation"] == "continue"
        assert rows[2]["relation"] == "new_document"
        assert "signature_break" in str(gap(rows[2]))

    def test_a_signature_on_the_documents_first_page_does_not_count(self):
        pages = [
            page(1, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(1, 3),
                 signed=True),
            page(2, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(2, 3)),
            page(3, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(3, 3),
                 signed=True),
            page(4, head=self.STICKY, dos_from="2025-07-29", dos_trust=5, page_type="Lab Report", printed=(1, 1)),
        ]
        rows = assign(pages)
        assert [r["relation"] for r in rows[1:]] == ["continue", "continue", "new_document"]
        assert "signature_break" not in str(gap(rows[1]))

    def test_a_lone_signature_page_joins_the_next_document_with_its_date(self):
        """2026-11-01 (user): a signature-only page is never a document of its own."""
        pages = [
            page(1, dos_from="2025-07-09", dos_trust=5, page_type="Medication Record", signed=True),
            page(2, dos_from="2025-07-29", dos_trust=3, page_type="Progress Note", page_subtype="Signature",
                 signed=True),
            page(3, "chief complaint: pain", dos_from="2025-07-29", dos_trust=3, page_type="Operative Note"),
            page(4, dos_from="2025-07-29", dos_trust=3, page_type="Operative Note"),
        ]
        rows = assign(pages)
        assert [r["relation"] for r in rows] == ["new_document", "new_document", "continue", "continue"]

    def test_a_lone_signature_page_closes_the_document_before_it(self):
        pages = [
            page(1, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note"),
            page(2, "chief complaint: pain", dos_from="2025-07-29", dos_trust=5, page_type="Progress Note",
                 page_subtype="Signature"),
            page(3, dos_from="2025-08-02", dos_trust=5, page_type="Lab Report"),
        ]
        rows = assign(pages)
        assert [r["relation"] for r in rows] == ["new_document", "continue", "new_document"]

    def test_a_section_continued_at_the_top_joins_over_a_date_and_a_visit_heading(self):
        """2026-11-01 (user): 'Medication List (continued)' at the top of a page whose
        section the page before has: one document, whatever the dates; a signed page
        still ends it."""
        pages = [
            page(1, "medication list\naspirin", dos_from="2019-04-26", dos_trust=5, page_type="Medication Record"),
            page(2, "chief complaint: pain", head="medication list (continued)", dos_from="2023-12-01",
                 dos_trust=5, page_type="Progress Note"),
            page(3, "notes", head="medication list (continued)", dos_from="2025-04-16", dos_trust=5,
                 page_type="Medication Record", signed=True),
            page(4, head="medication list (continued)", dos_from="2025-04-16", dos_trust=5,
                 page_type="Medication Record"),
        ]
        rows = assign(pages)
        assert [r["relation"] for r in rows] == ["new_document", "continue", "continue", "new_document"]
        assert "section_continued" in str(gap(rows[1]))

    def test_a_continued_name_the_page_before_lacks_does_not_join(self):
        pages = [page(1, "allergies", dos_from="2019-04-26", dos_trust=5, page_type="Medication Record"),
                 page(2, head="problem list (continued)", dos_from="2023-12-01", dos_trust=5,
                      page_type="Problem List")]
        assert assign(pages)[1]["relation"] == "new_document"

    def test_the_switch_turns_it_off(self):
        rules = dataclasses.replace(RULES, signature_breaks=False)
        pages = [page(1, dos_from="2025-07-29", dos_trust=5, signed=True, printed=(1, 2)),
                 page(2, dos_from="2025-07-29", dos_trust=5, printed=(2, 2))]
        assert assign(pages, rules)[1]["relation"] == "continue"

    def test_a_summary_page_stands_apart_but_summary_pages_join(self):
        pages = [
            page(1, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(1, 4)),
            page(2, dos_from="2025-07-29", dos_trust=5, page_type="Clinical Summary", printed=(2, 4)),
            page(3, dos_from="2025-07-29", dos_trust=5, page_type="Clinical Summary", printed=(3, 4)),
            page(4, dos_from="2025-07-29", dos_trust=5, page_type="Progress Note", printed=(4, 4)),
        ]
        rows = assign(pages)
        assert [r["relation"] for r in rows[1:]] == ["new_document", "continue", "new_document"]
