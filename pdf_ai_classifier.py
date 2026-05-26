from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


WORKSPACE = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE / "ai_output"
PUBLIC_DIR = WORKSPACE / "public"
MODEL_PATH = OUTPUT_DIR / "manual_classifier_model.joblib"
ASSISTANT_PATH = OUTPUT_DIR / "manual_assistant_index.json"
DATASET_PATH = OUTPUT_DIR / "training_dataset.csv"
ANALYTICS_PATH = OUTPUT_DIR / "analytics.json"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"


CATEGORIES = {
    "safety": [
        "safety", "danger", "warning", "caution", "risk", "hazard", "ppe",
        "emergency", "fire", "shock", "burn", "injury", "protection",
    ],
    "installation": [
        "installation", "install", "mounting", "fixing", "anchoring", "leveling",
        "transport", "handling", "packaging", "location", "clearance", "housing",
    ],
    "operation": [
        "operation", "start", "stop", "running", "commissioning", "preliminary",
        "check", "control panel", "on/off", "heating mode", "cooling mode",
    ],
    "maintenance_service": [
        "maintenance", "service", "replace", "replacement", "clean", "filter",
        "oil", "anode", "impeller", "belt", "monthly", "yearly", "inspection",
    ],
    "troubleshooting_repair": [
        "troubleshooting", "fault", "failure", "problem", "repair", "cause",
        "remedy", "does not start", "low voltage", "high voltage", "smoke",
    ],
    "technical_specs": [
        "technical", "specification", "features", "capacity", "voltage",
        "frequency", "power", "dimensions", "weight", "model", "flow", "btu",
    ],
    "electrical_wiring": [
        "electrical", "electric", "wiring", "terminal", "circuit breaker",
        "ground", "battery", "fuse", "connector", "inverter", "voltage",
    ],
    "cooling_water": [
        "cooling", "seawater", "sea water", "fresh water", "glycol", "pump",
        "heat exchanger", "condenser", "flow", "water circuit", "chiller",
    ],
    "fuel_system": [
        "fuel", "diesel", "fuel filter", "fuel pump", "tank", "bleeding",
        "deaeration", "combustion", "injection",
    ],
    "alarms_controls": [
        "alarm", "display", "sensor", "probe", "controller", "menu", "setting",
        "digital input", "eco", "saved alarm", "flow alarm", "code",
    ],
    "spare_parts": [
        "spare", "parts", "kit", "code", "cartridge", "belt", "impeller",
        "anode", "recommended spare",
    ],
    "warranty_disposal": [
        "warranty", "responsibility", "liability", "disposal", "recovery",
        "refrigerant", "gas recovery", "guarantee",
    ],
}


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "if", "in", "is", "it", "its", "may", "must", "not", "of", "on",
    "or", "shall", "should", "the", "this", "to", "use", "used", "with",
    "without", "you", "your", "page", "manual", "unit", "machine",
}


@dataclass
class Example:
    pdf: str
    page: int
    category: str
    confidence: float
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_/.-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def chunk_text(text: str, max_words: int = 140) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
    return chunks


def label_text(text: str) -> tuple[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for category, keywords in CATEGORIES.items():
        score = 0.0
        for keyword in keywords:
            if " " in keyword:
                score += 2.5 * lowered.count(keyword)
            else:
                score += len(re.findall(rf"\b{re.escape(keyword)}\b", lowered))
        scores[category] = score

    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values())
    if best_score <= 0:
        return "general_information", 0.2
    return best_category, round(best_score / max(total, 1.0), 4)


def extract_examples(pdf_dir: Path) -> list[Example]:
    examples: list[Example] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        if pdf_path.name.lower().endswith("_report.pdf") or "troubleshooting_model_report" in pdf_path.name.lower():
            continue
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception:
                text = ""
            for chunk in chunk_text(text):
                category, confidence = label_text(chunk)
                examples.append(
                    Example(
                        pdf=pdf_path.name,
                        page=page_index,
                        category=category,
                        confidence=confidence,
                        text=chunk,
                    )
                )
    return examples


class NaiveBayesTextClassifier:
    def __init__(self) -> None:
        self.class_doc_counts: Counter[str] = Counter()
        self.class_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, rows: Iterable[Example]) -> None:
        for row in rows:
            tokens = tokenize(row.text)
            if not tokens:
                continue
            self.class_doc_counts[row.category] += 1
            self.class_token_counts[row.category].update(tokens)
            self.class_total_tokens[row.category] += len(tokens)
            self.vocabulary.update(tokens)

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        if not tokens or not self.class_doc_counts:
            return {}

        total_docs = sum(self.class_doc_counts.values())
        vocab_size = max(len(self.vocabulary), 1)
        log_scores: dict[str, float] = {}

        for category, doc_count in self.class_doc_counts.items():
            log_prob = math.log(doc_count / total_docs)
            denom = self.class_total_tokens[category] + vocab_size
            token_counts = self.class_token_counts[category]
            for token in tokens:
                log_prob += math.log((token_counts[token] + 1) / denom)
            log_scores[category] = log_prob

        max_log = max(log_scores.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in log_scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in sorted(exp_scores.items(), key=lambda item: item[1], reverse=True)}

    def predict(self, text: str) -> tuple[str, float]:
        probabilities = self.predict_proba(text)
        if not probabilities:
            return "unknown", 0.0
        category, probability = next(iter(probabilities.items()))
        return category, probability

    def to_dict(self) -> dict:
        return {
            "class_doc_counts": dict(self.class_doc_counts),
            "class_token_counts": {k: dict(v) for k, v in self.class_token_counts.items()},
            "class_total_tokens": dict(self.class_total_tokens),
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NaiveBayesTextClassifier":
        model = cls()
        model.class_doc_counts = Counter(payload["class_doc_counts"])
        model.class_token_counts = defaultdict(Counter, {k: Counter(v) for k, v in payload["class_token_counts"].items()})
        model.class_total_tokens = Counter(payload["class_total_tokens"])
        model.vocabulary = set(payload["vocabulary"])
        return model


class SklearnTextClassifier:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOPWORDS),
            ngram_range=(1, 2),
            min_df=1,
            max_features=12000,
        )
        self.classifier = MultinomialNB(alpha=0.35)
        self.labels: list[str] = []
        self.category_terms: dict[str, list[str]] = {}

    def fit(self, rows: Iterable[Example]) -> None:
        rows = list(rows)
        texts = [row.text for row in rows]
        labels = [row.category for row in rows]
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        self.labels = list(self.classifier.classes_)
        self.category_terms = self._extract_category_terms()

    def _extract_category_terms(self, limit: int = 18) -> dict[str, list[str]]:
        feature_names = self.vectorizer.get_feature_names_out()
        terms: dict[str, list[str]] = {}
        for class_index, label in enumerate(self.classifier.classes_):
            weights = self.classifier.feature_log_prob_[class_index]
            top_indexes = weights.argsort()[-limit:][::-1]
            terms[label] = [feature_names[index] for index in top_indexes]
        return terms

    def predict_proba(self, text: str) -> dict[str, float]:
        matrix = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(matrix)[0]
        pairs = sorted(zip(self.classifier.classes_, probabilities), key=lambda item: item[1], reverse=True)
        return {label: float(probability) for label, probability in pairs}

    def predict(self, text: str) -> tuple[str, float]:
        probabilities = self.predict_proba(text)
        category, probability = next(iter(probabilities.items()))
        return category, probability

    def to_dict(self) -> dict:
        return {
            "library": "scikit-learn",
            "model": "sklearn.naive_bayes.MultinomialNB",
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "labels": self.labels,
            "category_terms": self.category_terms,
        }

    def to_joblib_payload(self) -> dict:
        return {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.labels,
            "category_terms": self.category_terms,
        }

    @classmethod
    def from_joblib_payload(cls, payload: dict) -> "SklearnTextClassifier":
        model = cls()
        model.vectorizer = payload["vectorizer"]
        model.classifier = payload["classifier"]
        model.labels = payload["labels"]
        model.category_terms = payload["category_terms"]
        return model


def train_test_split(examples: list[Example], test_every: int = 5) -> tuple[list[Example], list[Example]]:
    train, test = [], []
    for index, row in enumerate(examples):
        if index % test_every == 0:
            test.append(row)
        else:
            train.append(row)
    return train, test


def evaluate(model: NaiveBayesTextClassifier, test_rows: list[Example]) -> dict:
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in test_rows:
        predicted, _ = model.predict(row.text)
        if predicted == row.category:
            correct += 1
        confusion[row.category][predicted] += 1
    accuracy = correct / max(len(test_rows), 1)
    return {
        "accuracy": round(accuracy, 4),
        "test_examples": len(test_rows),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def write_dataset(examples: list[Example]) -> None:
    with DATASET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdf", "page", "category", "confidence", "text"])
        writer.writeheader()
        for row in examples:
            writer.writerow(row.__dict__)


def read_dataset() -> list[Example]:
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as handle:
        return [
            Example(
                pdf=row["pdf"],
                page=int(row["page"]),
                category=row["category"],
                confidence=float(row["confidence"]),
                text=row["text"],
            )
            for row in csv.DictReader(handle)
        ]


def extract_facts(examples: list[Example]) -> dict:
    all_text = "\n".join(row.text for row in examples)
    intervals = sorted(set(re.findall(r"\b(?:daily|weekly|monthly|yearly|annually|every\s+\w+\s+months|once\s+a\s+\w+|at\s+least\s+once\s+a\s+\w+)\b", all_text, flags=re.I)))
    alarm_codes = sorted(set(re.findall(r"\b(?:E[1-5]|F[1-4]|B1|C1)\b", all_text)))
    service_terms = [
        "filter", "oil", "anode", "impeller", "belt", "pump", "sensor",
        "battery", "fuel", "seawater", "glycol", "condensate", "valve",
    ]
    service_counts = {term: len(re.findall(rf"\b{term}\b", all_text, flags=re.I)) for term in service_terms}
    model_mentions = sorted(set(re.findall(r"\b(?:VS\s*22\s*LOW|CU50VFD|CU70VFD|AH1|AH3|AH5|AH7|AH9|Yanmar\s+4TNV88)\b", all_text, flags=re.I)))
    page_counts = Counter(row.pdf for row in examples)
    category_counts = Counter(row.category for row in examples)
    pdf_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in examples:
        pdf_category_counts[row.pdf][row.category] += 1

    return {
        "document_chunks": dict(page_counts),
        "category_counts": dict(category_counts),
        "pdf_category_counts": {k: dict(v) for k, v in pdf_category_counts.items()},
        "maintenance_intervals": intervals[:40],
        "alarm_codes": alarm_codes,
        "service_term_counts": service_counts,
        "model_mentions": model_mentions,
    }


def top_terms(model: NaiveBayesTextClassifier, limit: int = 18) -> dict[str, list[str]]:
    if hasattr(model, "category_terms"):
        return {k: v[:limit] for k, v in model.category_terms.items()}
    result: dict[str, list[str]] = {}
    for category, counts in model.class_token_counts.items():
        result[category] = [term for term, _ in counts.most_common(limit)]
    return result


def build_assistant_index(examples: list[Example], model: NaiveBayesTextClassifier) -> dict:
    chunks = []
    document_frequency: Counter[str] = Counter()
    for index, row in enumerate(examples):
        tokens = tokenize(row.text)
        unique_tokens = set(tokens)
        document_frequency.update(unique_tokens)
        chunks.append({
            "id": index,
            "pdf": row.pdf,
            "page": row.page,
            "category": row.category,
            "text": row.text,
            "tokens": tokens,
            "length": len(tokens),
        })

    return {
        "chunks": chunks,
        "document_frequency": dict(document_frequency),
        "average_length": sum(chunk["length"] for chunk in chunks) / max(len(chunks), 1),
        "total_chunks": len(chunks),
        "categories": CATEGORIES,
        "classifier": model.to_dict(),
        "top_terms": top_terms(model),
    }


def build_dashboard(analytics: dict, evaluation: dict, assistant_index: dict) -> str:
    slim_index = {
        "chunks": [
            {k: chunk[k] for k in ["id", "pdf", "page", "category", "text", "tokens", "length"]}
            for chunk in assistant_index["chunks"]
        ],
        "document_frequency": assistant_index["document_frequency"],
        "average_length": assistant_index["average_length"],
        "total_chunks": assistant_index["total_chunks"],
        "categories": assistant_index["categories"],
        "top_terms": assistant_index["top_terms"],
    }
    data_json = json.dumps({"analytics": analytics, "evaluation": evaluation, "assistant": slim_index}, ensure_ascii=False)
    category_rows = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{count}</td></tr>"
        for category, count in sorted(analytics["category_counts"].items(), key=lambda item: item[1], reverse=True)
    )
    service_rows = "".join(
        f"<tr><td>{html.escape(term)}</td><td>{count}</td></tr>"
        for term, count in sorted(analytics["service_term_counts"].items(), key=lambda item: item[1], reverse=True)
    )
    doc_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in analytics["document_chunks"].items()
    )
    intervals = "".join(f"<li>{html.escape(item)}</li>" for item in analytics["maintenance_intervals"])
    alarms = ", ".join(html.escape(code) for code in analytics["alarm_codes"]) or "None detected"
    models = ", ".join(html.escape(model) for model in analytics["model_mentions"]) or "None detected"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yacht Manual AI Troubleshooter</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #172033; }}
    header {{ background: #123047; color: white; padding: 22px 32px; }}
    main {{ padding: 24px 32px 40px; max-width: 1320px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; }}
    .panel {{ background: white; border: 1px solid #dce3ee; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
    .wide {{ grid-column: span 2; }}
    .full {{ grid-column: 1 / -1; }}
    .metric {{ font-size: 30px; font-weight: 700; margin-top: 6px; }}
    .label {{ color: #5d6b7d; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    td, th {{ border-bottom: 1px solid #edf1f6; padding: 8px 6px; text-align: left; }}
    th {{ color: #506070; }}
    canvas {{ width: 100%; height: 250px; }}
    textarea {{ width: 100%; min-height: 125px; box-sizing: border-box; border: 1px solid #cfd8e6; border-radius: 6px; padding: 10px; font-size: 15px; }}
    button {{ background: #1b6ca8; color: white; border: 0; border-radius: 6px; padding: 9px 13px; cursor: pointer; }}
    .answer h3 {{ margin: 16px 0 8px; }}
    .answer li {{ margin: 6px 0; }}
    .source {{ color: #4f6072; font-size: 13px; margin-top: 5px; }}
    .pill {{ display: inline-block; background: #edf4fb; color: #174d75; border: 1px solid #cfe1f0; padding: 3px 8px; border-radius: 999px; margin: 2px; font-size: 12px; }}
    .muted {{ color: #627083; }}
    .split {{ display: grid; grid-template-columns: 1.4fr .9fr; gap: 14px; }}
    @media (max-width: 1000px) {{ .split {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .wide {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Yacht Manual AI Troubleshooter</h1>
    <div>Local assistant trained from the 4 PDF manuals for AC, generator, service, repair, and fault guidance.</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="label">Manual chunks indexed</div><div class="metric">{sum(analytics["category_counts"].values())}</div></div>
      <div class="panel"><div class="label">Categories</div><div class="metric">{len(analytics["category_counts"])}</div></div>
      <div class="panel"><div class="label">Classifier test accuracy</div><div class="metric">{evaluation["accuracy"] * 100:.1f}%</div></div>
      <div class="panel"><div class="label">Alarm codes found</div><div class="metric">{len(analytics["alarm_codes"])}</div></div>
      <div class="panel full">
        <div class="split">
          <div>
            <h2>Describe The Problem</h2>
            <textarea id="inputText">The yacht air conditioning has low pressure and the fresh water flow alarm appears. What could be the reason and how can I solve it?</textarea>
            <p><button onclick="solveProblem()">Resolve Issue</button></p>
            <div id="answer" class="answer"></div>
          </div>
          <div>
            <h2>What This AI Uses</h2>
            <p class="muted">The assistant combines a text classifier with manual retrieval. It searches the full indexed PDF content, then builds likely causes and actions from matching manual passages.</p>
            <div id="routing"></div>
          </div>
        </div>
      </div>
      <div class="panel wide"><h2>Category Distribution</h2><canvas id="categoryChart"></canvas></div>
      <div class="panel wide"><h2>Service Term Frequency</h2><canvas id="serviceChart"></canvas></div>
      <div class="panel wide"><h2>Classification Categories</h2><table><tr><th>Category</th><th>Chunks</th></tr>{category_rows}</table></div>
      <div class="panel wide"><h2>Service / Repair Signals</h2><table><tr><th>Term</th><th>Mentions</th></tr>{service_rows}</table></div>
      <div class="panel wide"><h2>Documents</h2><table><tr><th>PDF</th><th>Training chunks</th></tr>{doc_rows}</table></div>
      <div class="panel wide"><h2>Detected Maintenance Intervals</h2><ul>{intervals}</ul></div>
      <div class="panel full"><h2>Detected Equipment Models</h2><p>{models}</p><h2>Detected Alarm Codes</h2><p>{alarms}</p></div>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('dashboard-data').textContent);
    const categories = payload.analytics.category_counts;
    const services = payload.analytics.service_term_counts;
    const assistant = payload.assistant;

    function drawBarChart(canvasId, source, limit) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = 260 * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      const entries = Object.entries(source).sort((a, b) => b[1] - a[1]).slice(0, limit);
      const max = Math.max(...entries.map(x => x[1]), 1);
      ctx.clearRect(0, 0, rect.width, 260);
      entries.forEach((entry, i) => {{
        const y = 18 + i * 28;
        const width = (entry[1] / max) * (rect.width - 190);
        ctx.fillStyle = '#1b6ca8';
        ctx.fillRect(145, y, width, 17);
        ctx.fillStyle = '#172033';
        ctx.font = '12px Arial';
        ctx.fillText(entry[0], 4, y + 13);
        ctx.fillText(entry[1], 153 + width, y + 13);
      }});
    }}

    const stopwords = new Set({json.dumps(sorted(STOPWORDS))});
    function tokens(text) {{
      return (text.toLowerCase().match(/[a-z][a-z0-9_/.-]{{2,}}/g) || []).filter(t => !stopwords.has(t));
    }}

    function routeCategory(text) {{
      const lower = text.toLowerCase();
      let best = ['general_information', 0];
      Object.entries(assistant.categories).forEach(([category, words]) => {{
        let score = 0;
        words.forEach(word => {{ if (lower.includes(word.toLowerCase())) score += word.includes(' ') ? 2.5 : 1; }});
        if (score > best[1]) best = [category, score];
      }});
      return best[0];
    }}

    function bm25Search(query, limit = 8) {{
      const qTokens = tokens(query);
      const qSet = [...new Set(qTokens)];
      const totalDocs = assistant.total_chunks;
      const avgLen = assistant.average_length || 1;
      const k1 = 1.5;
      const b = 0.75;
      const routed = routeCategory(query);
      return assistant.chunks.map(chunk => {{
        const counts = {{}};
        chunk.tokens.forEach(t => counts[t] = (counts[t] || 0) + 1);
        let score = 0;
        qSet.forEach(term => {{
          const tf = counts[term] || 0;
          if (!tf) return;
          const df = assistant.document_frequency[term] || 0;
          const idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5));
          score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * chunk.length / avgLen)));
        }});
        if (chunk.category === routed) score *= 1.18;
        return {{...chunk, score}};
      }}).filter(x => x.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
    }}

    function sentenceSplit(text) {{
      return text.split(/(?<=[.!?])\\s+/).map(s => s.trim()).filter(s => s.length > 35);
    }}

    function pickSentences(results, words, limit) {{
      const needles = words.map(w => w.toLowerCase());
      const picked = [];
      results.forEach(result => {{
        sentenceSplit(result.text).forEach(sentence => {{
          const lower = sentence.toLowerCase();
          if (needles.some(word => lower.includes(word)) && !picked.some(x => x.text === sentence)) {{
            picked.push({{text: sentence, pdf: result.pdf, page: result.page}});
          }}
        }});
      }});
      return picked.slice(0, limit);
    }}

    function fallbackCauses(query, category) {{
      const lower = query.toLowerCase();
      const causes = [];
      if (lower.includes('low pressure') || lower.includes('flow alarm') || lower.includes('water flow')) {{
        causes.push('Fresh-water circulation may be missing or too low. The Frigomar chiller manual says the flow sensor status shows 0 when water is circulating and 1 when there is no circulation.');
        causes.push('Air may still be trapped in the fresh-water circuit, especially after filling or service.');
        causes.push('The fresh-water filter/strainer may be dirty, or a service/manifold valve may be closed or not balanced.');
        causes.push('The pump may be incorrectly installed, undersized, dry-running, or unable to move enough water.');
      }}
      if (lower.includes('air condition') || lower.includes('conditioning') || lower.includes('chiller')) {{
        causes.push('For the Frigomar chiller/fan-coil system, the manuals repeatedly point to water flow, strainers, pumps, valves, bleeding, sensors, and correct operating limits.');
      }}
      if (!causes.length) causes.push(`The issue appears closest to ${{category.replaceAll('_', ' ')}}. Check the matched manual sources below before replacing parts.`);
      return causes;
    }}

    function domainActions(query) {{
      const lower = query.toLowerCase();
      if ((lower.includes('air condition') || lower.includes('chiller') || lower.includes('fan coil')) &&
          (lower.includes('low pressure') || lower.includes('flow alarm') || lower.includes('fresh water'))) {{
        return [
          'Open the chiller display running-status page and check the fresh-water flow sensor. Status 0 means circulation is present; status 1 means no circulation.',
          'Confirm the fresh-water pump is running and that the system has been fully bled. If bleeding takes more than 10 minutes, restart the fresh-water pump and continue bleeding.',
          'Check that all fresh-water service valves, manifold valves, and fan-coil valves are open and balanced.',
          'Inspect and clean the fresh-water filter/strainer. Repeat cleaning after a few hours if the filter becomes dirty again.',
          'Check for water leaks along the fresh-water circuit and verify system pressure with the pump off and valves open.',
          'Check seawater circulation separately: clean the seawater strainer, verify seawater pump operation, and confirm water flow is within the manual range for CU50VFD or CU70VFD.',
          'Do not keep running the compressor with poor water circulation or active pressure/flow alarms. If the alarm remains, stop the unit and use qualified Frigomar service support.'
        ];
      }}
      return [];
    }}

    function renderApiAnswer(data) {{
      const reasons = (data.possible_reasons || []).map(item => {{
        if (typeof item === 'string') return `<li>${{escapeHtml(item)}}</li>`;
        return `<li><strong>${{escapeHtml(item.reason || 'Possible cause')}}</strong><br>${{escapeHtml(item.detail || '')}}<div class="source">${{escapeHtml(item.evidence || '')}}</div></li>`;
      }}).join('');
      const steps = (data.diagnostic_steps || data.suggested_actions || []).map(item => {{
        if (typeof item === 'string') return `<li>${{escapeHtml(item)}}</li>`;
        return `<li><strong>${{escapeHtml(item.step || 'Check')}}</strong><br><span class="muted">${{escapeHtml(item.why || '')}}</span></li>`;
      }}).join('');
      const sources = (data.manual_sources || []).map(r => `<li><strong>${{escapeHtml(r.pdf)}}</strong>, page ${{r.page}} <span class="muted">(score ${{Number(r.score || 0).toFixed(2)}}, ${{escapeHtml((r.category || '').replaceAll('_', ' '))}})</span><div class="source">${{escapeHtml((r.excerpt || '').slice(0, 500))}}...</div></li>`).join('');
      const notes = (data.safety_notes || []).map(n => `<li>${{escapeHtml(n)}}</li>`).join('');
      document.getElementById('answer').innerHTML = `
        <h3>Answer Summary</h3>
        <p>${{escapeHtml(data.answer_summary || 'No summary returned.')}}</p>
        <h3>Likely Issue Area</h3>
        <p><span class="pill">${{escapeHtml((data.detected_area || 'unknown').replaceAll('_', ' '))}}</span> <span class="pill">${{escapeHtml((data.issue_type || 'general').replaceAll('_', ' '))}}</span></p>
        <h3>Possible Reasons</h3>
        <ul>${{reasons}}</ul>
        <h3>Diagnostic Checks / Fixes</h3>
        <ol>${{steps}}</ol>
        <h3>Manual Evidence</h3>
        <ol>${{sources}}</ol>
        ${{notes ? `<h3>Safety Notes</h3><ul>${{notes}}</ul>` : ''}}
      `;
      document.getElementById('routing').innerHTML = `
        <p><strong>Detected area:</strong> ${{escapeHtml((data.detected_area || 'unknown').replaceAll('_', ' '))}}</p>
        <p><strong>Issue type:</strong> ${{escapeHtml((data.issue_type || 'general').replaceAll('_', ' '))}}</p>
        <p><strong>Answer mode:</strong> Python resolver API + manual retrieval</p>
        <p class="muted">${{escapeHtml(data.optional_gpt_note || '')}}</p>
      `;
    }}

    async function solveProblem() {{
      const query = document.getElementById('inputText').value;
      document.getElementById('answer').innerHTML = '<p class="muted">Resolving issue from manual index...</p>';
      try {{
        const response = await fetch('/api/resolve', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{query}})
        }});
        if (response.ok) {{
          renderApiAnswer(await response.json());
          return;
        }}
      }} catch (error) {{
        console.warn('Resolver API unavailable, falling back to browser resolver.', error);
      }}
      const routed = routeCategory(query);
      const results = bm25Search(query, 10);
      const causeSentences = pickSentences(results, ['cause', 'alarm', 'poor', 'dirty', 'clog', 'air', 'pressure', 'flow', 'pump', 'filter', 'strainer', 'valve'], 5);
      const actionSentences = pickSentences(results, ['check', 'replace', 'clean', 'bleed', 'open', 'reset', 'inspect', 'close', 'contact', 'run', 'drain'], 7);
      const domainFixes = domainActions(query);
      const causes = domainFixes.length ? fallbackCauses(query, routed) : (causeSentences.length ? causeSentences.map(x => x.text) : fallbackCauses(query, routed));
      const actions = domainFixes.length ? domainFixes : (actionSentences.length ? actionSentences.map(x => x.text) : [
        'Check the relevant water/fuel/electrical circuit first, then inspect filters, valves, pumps, sensors, and alarms according to the matched manual sources.',
        'If the displayed alarm remains after basic checks, contact an authorised service technician as stated in the manuals.'
      ]);
      const sources = results.slice(0, 5).map(r => `<li><strong>${{r.pdf}}</strong>, page ${{r.page}} <span class="muted">(score ${{r.score.toFixed(2)}}, ${{r.category.replaceAll('_', ' ')}})</span><div class="source">${{escapeHtml(r.text.slice(0, 360))}}...</div></li>`).join('');
      const causeHtml = causes.map(c => `<li>${{escapeHtml(c)}}</li>`).join('');
      const actionHtml = actions.map(a => `<li>${{escapeHtml(a)}}</li>`).join('');
      document.getElementById('answer').innerHTML = `
        <h3>Likely Issue Area</h3>
        <p><span class="pill">${{routed.replaceAll('_', ' ')}}</span></p>
        <h3>Possible Reasons</h3>
        <ul>${{causeHtml}}</ul>
        <h3>Suggested Checks / Fixes</h3>
        <ol>${{actionHtml}}</ol>
        <h3>Manual Evidence</h3>
        <ol>${{sources}}</ol>
        <p class="muted">Use this as technical guidance from the manuals. Electrical, refrigerant, pressure, and onboard safety work should be performed by qualified personnel.</p>
      `;
      document.getElementById('routing').innerHTML = `
        <p><strong>Detected area:</strong> ${{routed.replaceAll('_', ' ')}}</p>
        <p><strong>Matched passages:</strong> ${{results.length}}</p>
        <p><strong>Top terms for this area:</strong></p>
        <p>${{(assistant.top_terms[routed] || []).slice(0, 12).map(t => `<span class="pill">${{escapeHtml(t)}}</span>`).join('')}}</p>
      `;
    }}

    function escapeHtml(value) {{
      return value.replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
    }}

    drawBarChart('categoryChart', categories, 10);
    drawBarChart('serviceChart', services, 12);
    solveProblem();
  </script>
</body>
</html>"""


def build_all() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PUBLIC_DIR.mkdir(exist_ok=True)
    examples = extract_examples(WORKSPACE)
    write_dataset(examples)

    train_rows, test_rows = train_test_split(examples)
    model = SklearnTextClassifier()
    model.fit(train_rows)
    evaluation = evaluate(model, test_rows)

    dump(model.to_joblib_payload(), MODEL_PATH)
    assistant_index = build_assistant_index(examples, model)
    ASSISTANT_PATH.write_text(json.dumps(assistant_index, ensure_ascii=False), encoding="utf-8")
    analytics = extract_facts(examples)
    ANALYTICS_PATH.write_text(json.dumps({"analytics": analytics, "evaluation": evaluation}, indent=2, ensure_ascii=False), encoding="utf-8")
    dashboard_html = build_dashboard(analytics, evaluation, assistant_index)
    DASHBOARD_PATH.write_text(dashboard_html, encoding="utf-8")
    (PUBLIC_DIR / "index.html").write_text(dashboard_html, encoding="utf-8")
    return {
        "examples": len(examples),
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "evaluation": evaluation,
        "files": {
            "dataset": str(DATASET_PATH),
            "model": str(MODEL_PATH),
            "assistant_index": str(ASSISTANT_PATH),
            "analytics": str(ANALYTICS_PATH),
            "dashboard": str(DASHBOARD_PATH),
            "public_index": str(PUBLIC_DIR / "index.html"),
        },
    }


def load_model() -> NaiveBayesTextClassifier:
    payload = load(MODEL_PATH)
    if isinstance(payload, dict) and "vectorizer" in payload and "classifier" in payload:
        return SklearnTextClassifier.from_joblib_payload(payload)
    return payload


def main() -> None:
    result = build_all()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
