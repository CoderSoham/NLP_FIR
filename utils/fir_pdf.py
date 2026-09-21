"""The incident report, as a document rather than a dump of every field.

The previous version set a 10mm line height on every call, including the one
carrying a six-thousand-character transcript, so a long call produced forty
pages of double-spaced text with no headings, no page numbers and no way to
find anything. It also left out the entire LLM incident record -- the most
carefully produced part of the pipeline never reached the PDF at all.

Structured the way someone reading it actually needs it:

1. What happened, and how bad          -- the triage block, first page, always
2. What will hurt the responder        -- weapons, above everything else
3. What to do                          -- actions and resources
4. What it was inferred from           -- summary, entities, then the transcript
5. What not to trust                   -- warnings, disagreements, provenance

Imports no ML package, so it is testable in milliseconds.
"""
import os
import unicodedata

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# fpdf2's core fonts are Latin-1. Rather than drop what does not fit -- which
# deleted every curly apostrophe Whisper emits, turning "don't" into "dont",
# and silently removed every bullet character -- map the handful that actually
# occur onto characters that do.
TRANSLITERATE = {
    "‘": "'", "’": "'", "‚": ",", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", "•": "·", " ": " ",
    "′": "'", "″": '"', "€": "EUR", "₹": "Rs.",
}

# fpdf2's multi_cell leaves the cursor to the RIGHT of the block by default.
# Every wrapped paragraph therefore started where the previous one ended, which
# pushed "Priority:", the second weapon and every bullet after the first off
# into the right margin. Flowing text always returns to the left margin.
FLOW = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

BULLET = "\u00b7"          # middle dot: in Latin-1, unlike U+2022

PAGE_WIDTH = 210
MARGIN = 15
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

SEVERITY_FILL = {
    "critical": (183, 28, 28),
    "high": (216, 67, 21),
    "medium": (245, 166, 35),
    "low": (46, 125, 50),
}
INK = (17, 24, 39)
MUTED = (110, 118, 129)
RULE = (208, 213, 221)
BAND = (243, 244, 246)


def as_text(value):
    """Render a field value for a human.

    The extraction schema allows a list where a caller said several things --
    two injuries, two vehicles -- and printing the list put a Python repr in
    the report: ``['officer shot through the right arm']``.
    """
    if value is None or value == [] or value == {}:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(as_text(v) for v in value if as_text(v))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {as_text(v)}" for k, v in value.items()
                         if as_text(v))
    return str(value)


def sanitize(text):
    """Latin-1 safe text, transliterating rather than deleting."""
    if text is None:
        return ""
    text = str(text)
    for src, dst in TRANSLITERATE.items():
        text = text.replace(src, dst)
    # Decomposing catches the accented characters Latin-1 does not carry --
    # a name spelled with a combining mark keeps its letters.
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class FIRReport(FPDF):
    """Page furniture that has to appear on every page, including the ones
    the transcript spills onto."""

    def __init__(self, report_id, timestamp):
        super().__init__(format="A4")
        self.report_id = report_id
        self.timestamp = timestamp
        self.set_margins(MARGIN, 18, MARGIN)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*MUTED)
        self.cell(CONTENT_WIDTH / 2, 5, "EMERGENCY CALL INCIDENT REPORT")
        self.cell(CONTENT_WIDTH / 2, 5, sanitize(self.report_id[:12]),
                  align="R", **FLOW)
        self.ln(1)
        self.set_draw_color(*RULE)
        self.line(MARGIN, self.get_y(), PAGE_WIDTH - MARGIN, self.get_y())
        self.ln(4)
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(CONTENT_WIDTH, 4, sanitize(
            "Machine-generated from an audio recording. Every field is a "
            "model output and may be wrong."), **FLOW)
        self.cell(CONTENT_WIDTH / 2, 4, sanitize(f"Generated {self.timestamp}"))
        self.cell(CONTENT_WIDTH / 2, 4, f"Page {self.page_no()}/{{nb}}",
                  align="R", **FLOW)
        self.set_text_color(*INK)

    # ---- building blocks ---------------------------------------------------

    def heading(self, text):
        # A heading needs room for a couple of rows under it, or it reads as
        # an empty section and the content looks like it belongs to the next.
        if self.get_y() > 238:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 7, sanitize(text), **FLOW)
        self.ln(1)
        self.set_draw_color(*RULE)
        self.line(MARGIN, self.get_y() - 2, PAGE_WIDTH - MARGIN, self.get_y() - 2)
        self.ln(1)

    def body(self, text, size=10, height=5):
        self.set_font("Helvetica", "", size)
        self.multi_cell(CONTENT_WIDTH, height, sanitize(text), **FLOW)

    def note(self, text, size=8.5):
        self.set_font("Helvetica", "I", size)
        self.set_text_color(*MUTED)
        self.multi_cell(CONTENT_WIDTH, 4, sanitize(text), **FLOW)
        self.set_text_color(*INK)
        self.ln(1)

    def field(self, label, value, label_width=46):
        """One labelled row that wraps without breaking the label column."""
        if self.get_y() > 262:
            self.add_page()
        self.set_font("Helvetica", "B", 10)
        y = self.get_y()
        self.cell(label_width, 5, sanitize(label))
        self.set_font("Helvetica", "", 10)
        self.set_xy(MARGIN + label_width, y)
        self.multi_cell(CONTENT_WIDTH - label_width, 5,
                        sanitize(as_text(value) or "Not stated"), **FLOW)
        self.ln(0.5)

    def bullets(self, items, size=10):
        self.set_font("Helvetica", "", size)
        for item in items:
            if self.get_y() > 265:
                self.add_page()
                self.set_font("Helvetica", "", size)
            y = self.get_y()
            self.cell(5, 5, BULLET)
            self.set_xy(MARGIN + 5, y)
            self.multi_cell(CONTENT_WIDTH - 5, 5, sanitize(as_text(item)),
                            **FLOW)

    def callout(self, title, lines, fill=(183, 28, 28)):
        """A block that has to be seen before anything else on the page."""
        if self.get_y() > 230:
            self.add_page()
        self.set_fill_color(*fill)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(CONTENT_WIDTH, 8, sanitize(f"  {title}"), fill=True, **FLOW)
        self.set_text_color(*INK)
        self.set_fill_color(*BAND)
        self.set_font("Helvetica", "", 10)
        for line in lines:
            self.multi_cell(CONTENT_WIDTH, 5, sanitize(f"  {line}"),
                            fill=True, **FLOW)
        self.ln(3)


def _minutes(value):
    """Whole minutes. "5.0 minutes" reads as a measurement of something."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    return f"{number:g}"


def _mmss(seconds):
    """Durations as mm:ss. "9:50" is a length; "590.1s" is a measurement."""
    try:
        seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "unknown duration"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _triage_band(pdf, data, record):
    """Type, severity, priority and location, readable at arm's length."""
    severity = (record.get("severity") or data.get("severity") or "low").lower()
    incident = (record.get("incident_type") or data.get("emergency_type") or "unknown")
    fill = SEVERITY_FILL.get(severity, SEVERITY_FILL["low"])

    pdf.set_fill_color(*fill)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(CONTENT_WIDTH, 14,
             sanitize(f"  {incident.title()}  -  {severity.title()}"),
             fill=True, **FLOW)
    pdf.set_text_color(*INK)

    pdf.set_fill_color(*BAND)
    pdf.set_font("Helvetica", "", 10)
    response = data.get("response") or {}
    location = record.get("location") or data.get("probable_location")
    pdf.multi_cell(CONTENT_WIDTH, 6, sanitize(
        f"  Location: {location or 'not stated in the call'}"), fill=True, **FLOW)
    pdf.multi_cell(CONTENT_WIDTH, 6, sanitize(
        f"  Priority: {(response.get('priority') or 'unknown').title()}"
        f"   |   Protocol target: "
        f"{_minutes(response.get('estimated_response_time'))} minutes"),
        fill=True, **FLOW)

    # What recording this is, and how much of it was heard. Without this the
    # only identifying mark on the document was a truncated hex id in the page
    # header, and a partial analysis was disclosed nowhere near the verdict.
    duration = data.get("analysed_duration_s")
    source = data.get("source_duration_s")
    heard = f"{_mmss(duration)} analysed"
    if data.get("truncated") and source:
        heard += f" of a {_mmss(source)} recording"
    language = (data.get("language") or "").upper()
    pdf.multi_cell(CONTENT_WIDTH, 6, sanitize(
        f"  Call: {heard}"
        + (f"   |   Language: {language}" if language else "")
        + f"   |   Report {data.get('report_id', '')[:12]}"), fill=True, **FLOW)

    dispatch = data.get("dispatch") or {}
    line = _dispatch_line(dispatch)
    if line:
        pdf.multi_cell(CONTENT_WIDTH, 6, sanitize("  " + line),
                       fill=True, **FLOW)
    if dispatch.get("example_data"):
        # The one fabricated field on a page where everything else is
        # evidenced. It says so, in the band, next to the name it invented.
        pdf.multi_cell(CONTENT_WIDTH, 6, sanitize(
            "  PLACEHOLDER: the station registry is example data. These "
            "stations do not exist."), fill=True, **FLOW)
    pdf.ln(2)


def _dispatch_line(dispatch):
    """One line naming the station and saying how confident it is entitled to be."""
    station = dispatch.get("station")
    if not station:
        return ""
    eta, basis = dispatch.get("eta_min"), dispatch.get("basis")
    if basis == "nearest_by_distance":
        km = dispatch.get("distance_km")
        return (f"Nearest station: {station}"
                + (f" - {km} km away" if km is not None else "")
                + (f", about {eta} minutes by road" if eta else ""))
    if basis == "location_match":
        return (f"Nearest station: {station} (matched the word "
                f"'{dispatch.get('matched_on')}' in the location)"
                + (f". Its nominal response time is {eta} minutes -- a figure "
                   f"stored for this station, not an estimate for this call."
                   if eta else ""))
    return (f"Nearest station: {station} - default for this emergency type. "
            f"No location matched, so no ETA.")


def _weapons(pdf, record):
    """Above everything else, because it is what gets someone killed."""
    weapons = record.get("weapons") or []
    if not weapons:
        return
    lines = []
    for weapon in weapons:
        if isinstance(weapon, dict):
            item, quote = weapon.get("item"), weapon.get("quote")
            lines.append(f"{item}" + (f'  -  "{quote}"' if quote else ""))
        else:
            lines.append(str(weapon))
    heading = ("WEAPON MENTIONED IN THE CALL" if len(lines) == 1
               else "WEAPONS MENTIONED IN THE CALL")
    pdf.callout(heading, lines)
    caveat = ("Each entry is quoted from the transcript; a claim the model "
              "could not quote was dropped.")
    rejected = (record.get("_rejected") or {}).get("weapons")
    if rejected:
        caveat += " Dropped here: " + ", ".join(str(r) for r in rejected) + "."
    pdf.note(caveat)


def _reliability(pdf, data):
    """Everything a reader should distrust, collected in one place."""
    warnings = list(data.get("warnings") or [])
    disagreements = data.get("disagreements") or []
    if not (warnings or disagreements or data.get("truncated")):
        return

    lines = []
    if data.get("truncated"):
        lines.append(
            f"PARTIAL: only the first {data.get('analysed_duration_s')}s of a "
            f"{data.get('source_duration_s')}s recording was analysed.")
    for field in disagreements:
        lines.append(
            f"Disagreement on {str(field.get('field', '')).replace('_', ' ')}: "
            f"rules say '{field.get('classical')}', model says '{field.get('llm')}'.")
    for warning in warnings:
        lines.append(f"Stage did not run: {warning}")
    pdf.callout("READ WITH CARE", lines, fill=(216, 67, 21))


def _incident_record(pdf, record, meta):
    if not record:
        pdf.heading("Incident record")
        reason = (meta or {}).get("reason") or "no model backend configured"
        pdf.note(f"Not produced - {reason}. Everything above comes from the "
                 f"rule-based pipeline alone.")
        return

    pdf.heading("Incident record")
    pdf.field("What happened", record.get("summary"))
    pdf.field("Location", record.get("location"))
    pdf.field("Callback number", record.get("callback_number"))
    pdf.field("People involved", record.get("people_involved"))
    pdf.field("Injuries", record.get("injuries"))
    pdf.field("Vehicles", record.get("vehicles"))
    pdf.field("Suspect description", record.get("suspect_description"))

    uncertainties = record.get("uncertainties") or []
    if uncertainties:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "The model was unsure about", **FLOW)
        pdf.ln(1)
        pdf.bullets([str(u) for u in uncertainties], size=9.5)



def _actions(pdf, data):
    response = data.get("response") or {}
    suggestions = response.get("suggestions") or []
    resources = response.get("required_resources") or []
    if not (suggestions or resources):
        return

    pdf.heading("Recommended actions")
    if suggestions:
        pdf.bullets(suggestions)
    if resources:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Resources", **FLOW)
        pdf.ln(1)
        pdf.bullets([str(r).replace("_", " ").title() for r in resources], size=9.5)

    signals = response.get("signals") or {}
    if signals:
        pdf.ln(2)
        pdf.note("Some actions above were added because these words appear in "
                 "the transcript. They are keyword matches, not judgements: "
                 + "; ".join(f"{group}: {', '.join(words)}"
                             for group, words in signals.items()))


def _summary_and_entities(pdf, data, record):
    # The incident record already printed its summary as "What happened".
    # Printing the same sentence again under its own heading was the report
    # saying one thing twice, which is how a reader learns to skim.
    summary = data.get("summary")
    if summary and summary.strip() != (record.get("summary") or "").strip():
        pdf.heading("Summary")
        pdf.body(summary)
        if record:
            pdf.note("From the extractive summariser, which runs independently "
                     "of the model that produced the incident record.")

    entities = data.get("entities") or []
    if entities:
        pdf.heading("Named entities")
        # Grouped by type. A flat list of forty tags is unreadable, and the
        # question a reader has is "what addresses were mentioned", not "what
        # was the fourth entity".
        groups = {}
        for entity in entities:
            groups.setdefault(entity.get("label", "?"), []).append(
                entity.get("text", ""))
        for label, texts in sorted(groups.items()):
            pdf.field(label, ", ".join(_dedupe(texts)), label_width=28)


# Below this much remaining space, the transcript starts a fresh page. A
# transcript that begins four lines from the bottom is unreadable; one that
# leaves two thirds of a page blank looks like a bug.
TRANSCRIPT_MIN_SPACE_MM = 110


def _dedupe(texts):
    """Distinct mentions, in the order they appeared.

    Case- and punctuation-insensitive, because spaCy's spans keep whatever
    trailing period ended the sentence -- so "Best Auto Wash" and "Best Auto
    Wash." were listed as two separate organisations.
    """
    seen, unique = set(), []
    for text in texts:
        key = "".join(c for c in (text or "").lower() if c.isalnum())
        if key and key not in seen:
            seen.add(key)
            unique.append(text.strip(" .,;:"))
    return unique


def _transcript(pdf, data):
    if pdf.get_y() > 297 - TRANSCRIPT_MIN_SPACE_MM:
        pdf.add_page()
    pdf.heading("Transcript")
    asr = data.get("asr") or {}
    provenance = [p for p in (asr.get("model"), asr.get("backend"),
                              asr.get("device")) if p]
    if provenance:
        pdf.note("Transcribed by " + " / ".join(str(p) for p in provenance)
                 + ". Speech recognition on telephone audio misreads names and "
                   "numbers; check anything that will be acted on.")

    segments = data.get("asr_segments") or []
    if segments:
        # Timestamps make a passage findable in the recording, which is the
        # only reason anyone opens the transcript section of a report.
        pdf.set_font("Helvetica", "", 9)
        for segment in segments:
            start = segment.get("start")
            stamp = (f"{int(start // 60):02d}:{int(start % 60):02d}"
                     if isinstance(start, (int, float)) else "--:--")
            y = pdf.get_y()
            if y > 265:
                pdf.add_page()
                y = pdf.get_y()
            pdf.set_text_color(*MUTED)
            pdf.cell(14, 4.5, stamp)
            pdf.set_text_color(*INK)
            pdf.set_xy(MARGIN + 14, y)
            pdf.multi_cell(CONTENT_WIDTH - 14, 4.5,
                           sanitize(segment.get("text", "").strip()), **FLOW)
    else:
        pdf.body(data.get("transcription") or "", size=9, height=4.5)

    translated = data.get("translated_text")
    if translated:
        pdf.heading("English translation")
        pdf.body(translated, size=9, height=4.5)


def _provenance(pdf, data):
    pdf.heading("How this was produced")
    meta = data.get("llm_meta") or {}
    asr = data.get("asr") or {}
    pdf.field("Speech recognition", asr.get("model"), label_width=46)
    if meta.get("status") == "ok":
        pdf.field("Incident record", meta.get("model"), label_width=46)
        if meta.get("failed_over_from"):
            pdf.field("Failed over", sanitize(
                f"{meta['failed_over_from']} did not answer; "
                f"{meta.get('backend')} was used instead."), label_width=46)
    else:
        pdf.field("Incident record",
                  f"not produced - {meta.get('reason', 'unavailable')}",
                  label_width=46)
    pdf.field("Audio analysed",
              _mmss(data.get("analysed_duration_s"))
              + (f" of {_mmss(data.get('source_duration_s'))}"
                 if data.get("truncated") else ""), label_width=46)
    # Absent, not neutral. The affect models are skipped when the record came
    # from the extraction model, and printing "NEU 0.50" for a reading nobody
    # took would be a measurement of nothing.
    sentiment = data.get("sentiment") or {}
    emotion = data.get("emotion") or {}
    if sentiment or emotion:
        pdf.field("Caller affect", sanitize(
            f"{sentiment.get('label', '?')} "
            f"({float(sentiment.get('score') or 0):.2f})   |   "
            f"{emotion.get('label', '?')} "
            f"({float(emotion.get('score') or 0):.2f})"), label_width=46)
        pdf.note("Affect is a weak signal on telephone audio and feeds the "
                 "severity score only as one term of four.")

    if data.get("second_opinion") is False:
        pdf.field("Second opinion", "not run - the extraction model classified "
                                    "this call and the local classifiers were "
                                    "skipped", label_width=46)


def generate(data, output_folder):
    """Write the report and return its path."""
    record = data.get("llm") or {}
    pdf = FIRReport(data.get("report_id", ""), data.get("timestamp", ""))
    pdf.alias_nb_pages()
    pdf.add_page()

    _triage_band(pdf, data, record)
    _weapons(pdf, record)
    _reliability(pdf, data)
    _incident_record(pdf, record, data.get("llm_meta"))
    _actions(pdf, data)
    _summary_and_entities(pdf, data, record)
    _provenance(pdf, data)
    _transcript(pdf, data)

    # One file per report. This used to write a single shared
    # `processed/fir_report.pdf`, so two concurrent callers overwrote each
    # other and /download_fir served whichever finished last.
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"fir_{data['report_id']}.pdf")
    pdf.output(out_path)
    return out_path
