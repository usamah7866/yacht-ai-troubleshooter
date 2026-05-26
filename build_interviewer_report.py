from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ai_output"
REPORT_PATH = ROOT / "Yacht_AI_Troubleshooting_Model_Report.pdf"


def load_data() -> dict:
    path = OUTPUT / "analytics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"analytics": {}, "evaluation": {"accuracy": 0, "test_examples": 0}}


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#18324a"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#52616f"),
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#18324a"),
            spaceBefore=11,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#202a36"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#4f5f6f"),
        ),
        "th": ParagraphStyle(
            "TH",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10.5,
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "TD",
            parent=base["BodyText"],
            fontSize=8.2,
            leading=10.8,
            textColor=colors.HexColor("#202a36"),
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18324a")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8e0ea")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
            ]
        )
    )
    return t


def note(text: str, st: dict[str, ParagraphStyle]):
    t = Table([[para(text, st["body"])]], colWidths=[17.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef5fb")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8d4ea")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7785"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "AI Manual Troubleshooting Prototype")
    canvas.drawRightString(19.2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> Path:
    payload = load_data()
    analytics = payload.get("analytics", {})
    evaluation = payload.get("evaluation", {})
    category_counts = analytics.get("category_counts", {})
    doc_chunks = analytics.get("document_chunks", {})

    total_examples = sum(category_counts.values())
    test_examples = evaluation.get("test_examples", 0)
    train_examples = max(total_examples - test_examples, 0)
    accuracy = evaluation.get("accuracy", 0) * 100

    st = styles()
    story = []

    story.append(para("AI Manual Troubleshooting Prototype", st["title"]))
    story.append(
        para(
            "Sample built from yacht generator and air-conditioning PDF manuals. The system trains a text model and provides a dashboard where a user can type a technical issue and receive detailed manual-based troubleshooting guidance.",
            st["subtitle"],
        )
    )
    story.append(
        table(
            [
                [para("Prepared by", st["th"]), para("Date", st["th"]), para("Project status", st["th"])],
                [para("Usama", st["td"]), para(date.today().strftime("%d %B %Y"), st["td"]), para("Working local prototype", st["td"])],
            ],
            [5.7 * cm, 5.7 * cm, 5.8 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        note(
            "<b>Purpose:</b> This report summarizes the sample I created for the requested AI system. It explains the data used, model training, testing result, and dashboard behavior in a simple flow.",
            st,
        )
    )

    story.append(para("1. Data Used", st["h1"]))
    story.append(
        para(
            "The prototype uses four PDF manuals placed in the project folder. These manuals cover yacht/marine equipment including a Mase VS 22 LOW generator and Frigomar air-conditioning/chiller/fan-coil systems.",
            st["body"],
        )
    )
    rows = [[para("PDF manual", st["th"]), para("Text chunks created", st["th"])]]
    for name, count in doc_chunks.items():
        rows.append([para(name, st["td"]), para(str(count), st["td"])])
    story.append(table(rows, [13.4 * cm, 3.8 * cm]))
    story.append(
        para(
            "The PDF text was extracted page by page, cleaned, and split into smaller chunks. These chunks became the base training data and also the searchable manual knowledge index.",
            st["body"],
        )
    )

    story.append(para("2. Model Training", st["h1"]))
    story.append(
        para(
            "I used a practical text classification model from scikit-learn. The model first converts manual text into numerical text features, then learns how different service topics are written in the manuals.",
            st["body"],
        )
    )
    story.append(
        table(
            [
                [para("Part", st["th"]), para("Used", st["th"]), para("Reason", st["th"])],
                [para("Text features", st["td"]), para("TfidfVectorizer", st["td"]), para("Turns manual text into useful word/phrase features.", st["td"])],
                [para("Classifier", st["td"]), para("MultinomialNB", st["td"]), para("Good baseline for text classification, fast, explainable, and reliable for small document datasets.", st["td"])],
                [para("Search layer", st["td"]), para("BM25-style retrieval", st["td"]), para("Finds the most relevant manual passages for the user problem.", st["td"])],
                [para("Answer layer", st["td"]), para("Python resolver API", st["td"]), para("Builds a detailed answer with summary, possible causes, diagnostic steps, safety notes, and PDF/page evidence.", st["td"])],
            ],
            [4.0 * cm, 4.6 * cm, 8.6 * cm],
        )
    )
    story.append(
        para(
            "The classifier detects the issue area, for example cooling water, fuel system, electrical wiring, alarms, spare parts, maintenance, or troubleshooting. The retrieval layer finds exact manual sections. The resolver then turns those passages into an ordered diagnostic answer instead of giving only generic suggestions.",
            st["body"],
        )
    )

    story.append(para("3. Training and Testing Result", st["h1"]))
    story.append(
        table(
            [
                [para("Metric", st["th"]), para("Result", st["th"])],
                [para("Total extracted examples", st["td"]), para(str(total_examples), st["td"])],
                [para("Training examples", st["td"]), para(str(train_examples), st["td"])],
                [para("Testing examples", st["td"]), para(str(test_examples), st["td"])],
                [para("Classifier test accuracy", st["td"]), para(f"{accuracy:.2f}%", st["td"])],
                [para("Trained model file", st["td"]), para("ai_output/manual_classifier_model.joblib", st["td"])],
                [para("Manual index file", st["td"]), para("ai_output/manual_assistant_index.json", st["td"])],
            ],
            [7.5 * cm, 9.7 * cm],
        )
    )
    story.append(
        para(
            "The current labels are automatically generated from the manuals, so the accuracy is a realistic prototype result. In a production version, the next step would be to add manually reviewed technician labels and real service-ticket examples.",
            st["body"],
        )
    )

    story.append(para("4. Dashboard", st["h1"]))
    story.append(
        para(
            "The dashboard is a local web page where the user writes the equipment problem in normal text. When served with python serve_dashboard.py, it calls the Python resolver API, detects the issue area, searches the manuals, and shows detailed possible causes, diagnostic checks/fixes, safety notes, and source evidence.",
            st["body"],
        )
    )
    story.append(
        table(
            [
                [para("Dashboard input", st["th"]), para("Dashboard output", st["th"])],
                [
                    para("The yacht air conditioning has low pressure and the fresh water flow alarm appears. What could be the reason and how can I solve it?", st["td"]),
                    para("Detected area: cooling_water<br/><br/><b>Detailed answer:</b> The manuals point first to a circulation problem, not immediately to replacing parts.<br/><br/><b>Possible reasons:</b> no/low fresh-water circulation, trapped air, dirty filter/strainer, closed or unbalanced valve, pump/seawater-flow issue, water leak, or incorrect circuit pressure.<br/><br/><b>Diagnostic checks:</b> check flow-sensor status, verify pump operation, bleed the system, clean filter/strainer, inspect valves/manifold balance, check leaks and pressure, then check seawater-side flow.", st["td"]),
                ],
            ],
            [7.0 * cm, 10.2 * cm],
        )
    )
    story.append(
        para(
            "For this example, the assistant found supporting evidence from the Frigomar CU50VFD/CU70VFD manual and the AH Series fan-coil manual, including pages related to fresh-water flow alarm, pump/flow checks, strainer/filter cleaning, manifold balancing, water-circuit bleeding, and water-pressure limits.",
            st["body"],
        )
    )
    story.append(
        note(
            "<b>Important run note:</b> For the improved detailed answers, the dashboard should be opened through the local server, for example python serve_dashboard.py then http://localhost:8765/dashboard.html. Opening dashboard.html directly as a file uses the weaker browser fallback.",
            st,
        )
    )

    story.append(para("5. Files Created", st["h1"]))
    story.append(
        table(
            [
                [para("File", st["th"]), para("Purpose", st["th"])],
                [para("pdf_ai_classifier.py", st["td"]), para("Main training/build script.", st["td"])],
                [para("resolve_issue.py", st["td"]), para("Tests the troubleshooting assistant from the command line.", st["td"])],
                [para("classify_text.py", st["td"]), para("Tests only the trained classifier.", st["td"])],
                [para("serve_dashboard.py", st["td"]), para("Runs the local dashboard server and resolver API.", st["td"])],
                [para("ai_output/dashboard.html", st["td"]), para("User-facing dashboard.", st["td"])],
                [para("ai_output/manual_classifier_model.joblib", st["td"]), para("Trained sklearn model.", st["td"])],
            ],
            [6.4 * cm, 10.8 * cm],
        )
    )

    story.append(para("6. Next Steps", st["h1"]))
    story.append(
        para(
            "This is a working sample. To make it production-ready, I would add more manuals, expert-labeled service examples, real fault tickets, better semantic search with embeddings, and optionally a GPT/LLM layer to write more natural answers while keeping every answer grounded in retrieved manual evidence and PDF/page references.",
            st["body"],
        )
    )
    story.append(
        note(
            "<b>Summary:</b> I created a sample AI troubleshooting system from the supplied PDFs. It trains a classifier, builds a searchable manual index, and provides a dashboard that helps answer yacht equipment issues using the manuals.",
            st,
        )
    )

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.6 * cm,
        title="AI Manual Troubleshooting Prototype Report",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return REPORT_PATH


if __name__ == "__main__":
    print(build_report())
