# PDF AI Classification Dashboard

This project trains a practical local troubleshooting assistant from the four marine equipment PDF manuals in this folder.

It does not use an external LLM. It combines two local AI/NLP models:

- A scikit-learn `TfidfVectorizer` + `MultinomialNB` text classifier to detect the issue area.
- A BM25-style retrieval model to search the full manual content and return evidence-backed causes and fixes.

This is more realistic than a classifier alone because service repair questions need exact manual evidence, not just a category label.

## Run In VS Code

1. Open VS Code.
2. Select `File > Open Folder...`.
3. Open this folder:

```text
C:\Users\usama\OneDrive\Desktop\yachats
```

4. Open the VS Code terminal:

```text
Terminal > New Terminal
```

5. Create a virtual environment:

```powershell
python -m venv .venv
```

6. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

7. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

8. Build the AI model, manual index, and dashboard:

```powershell
python pdf_ai_classifier.py
```

9. Start the dashboard server:

```powershell
python serve_dashboard.py
```

10. Open this in your browser:

```text
http://localhost:8765/dashboard.html
```

If port `8765` is already busy, run the server on another port:

```powershell
python serve_dashboard.py 8877
```

Then open:

```text
http://localhost:8877/dashboard.html
```

## VS Code Tasks

You can also run these from `Terminal > Run Task...`:

- `Create virtual environment`
- `Install dependencies`
- `Build AI model and dashboard`
- `Serve dashboard`
- `Classify sample text`
- `Resolve sample issue`

## Generated Files

Files are written to `ai_output/`:

- `training_dataset.csv` - extracted PDF chunks with service-oriented labels
- `manual_classifier_model.joblib` - trained sklearn classification model
- `manual_assistant_index.json` - searchable manual knowledge index
- `analytics.json` - category counts, service terms, alarms, intervals, model names
- `dashboard.html` - troubleshooting assistant dashboard

## Resolve A Service Issue

Use this when the user describes a problem and wants possible reasons and fixes:

```powershell
python resolve_issue.py "The yacht air conditioning has low pressure and the fresh water flow alarm appears. What could be the reason and how can I solve it?"
```

The output includes:

- detected issue area
- possible reasons
- detailed diagnostic checks / fixes
- manual evidence with PDF name and page number

## Classify New Text Only

Use this only if you want the category label:

```powershell
python classify_text.py "Replace the seawater pump impeller and inspect zinc anodes"
```

## View Dashboard

Open:

```text
ai_output/dashboard.html
```

Or serve it locally:

```powershell
python serve_dashboard.py
```

Then visit:

```text
http://localhost:8765/dashboard.html
```

## Classification Labels

The current labels are:

- `safety`
- `installation`
- `operation`
- `maintenance_service`
- `troubleshooting_repair`
- `technical_specs`
- `electrical_wiring`
- `cooling_water`
- `fuel_system`
- `alarms_controls`
- `spare_parts`
- `warranty_disposal`
- `general_information`

## Why This Model

A pure classification model is not enough for yacht troubleshooting. If someone asks, "AC pressure is low, what could be the reason?", classification can only say something like `cooling_water` or `alarms_controls`. It cannot reliably explain causes or give repair steps.

That is why this project uses a hybrid approach:

1. The Naive Bayes classifier routes the problem into the correct technical area.
2. The BM25 retrieval model searches every extracted manual chunk.
3. The assistant response is built from the most relevant manual passages.

This is the best practical architecture for the current stage because it is:

- local and independent
- fast on normal PDF/manual data
- explainable with page references
- easy to scale to many manuals
- safer than a generative-only model because it grounds answers in source manuals
- realistic for service, repair, maintenance, and troubleshooting workflows

Later, for a larger production version, the BM25 retrieval layer can be upgraded to vector embeddings plus an LLM answer generator. The current structure already prepares the data for that upgrade.

## Optional GPT Upgrade

The current prototype is local and source-grounded. A GPT model can be added later as an answer-writing layer:

1. Retrieve the most relevant PDF passages using `manual_assistant_index.json`.
2. Send only those passages plus the user's issue to GPT.
3. Instruct GPT to answer only from the retrieved evidence and include PDF/page references.

This would improve wording and reasoning, but it should not replace retrieval. The manuals must remain the source of truth.

These are realistic for service, repair, yacht maintenance, and equipment-support workflows.
