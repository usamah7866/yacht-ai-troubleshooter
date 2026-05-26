# GitHub + Vercel Deployment

This project is prepared for deployment as:

- static dashboard: `public/index.html`
- Python serverless endpoint: `api/resolve.py`
- manual retrieval data: `ai_output/manual_assistant_index.json`

## Local Rebuild

```powershell
python pdf_ai_classifier.py
```

This regenerates:

- `ai_output/dashboard.html`
- `public/index.html`
- `ai_output/manual_assistant_index.json`
- `ai_output/manual_classifier_model.joblib`

## Local Test

```powershell
python serve_dashboard.py
```

Open:

```text
http://localhost:8765/dashboard.html
```

If the port is busy:

```powershell
python serve_dashboard.py 8877
```

## GitHub

Create a new **private** GitHub repository, then push this project:

```powershell
git init
git add .
git commit -m "Initial AI troubleshooting prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Vercel

In Vercel:

1. Click `Add New... > Project`.
2. Import the private GitHub repository.
3. Framework preset: `Other`.
4. Build command: leave empty.
5. Output directory: leave empty.
6. Deploy.

The site root `/` serves `public/index.html`.

The resolver API is:

```text
/api/resolve
```

## Runtime Notes

- `.vercelignore` excludes source PDFs and the report PDF from deployment to keep the Vercel upload small.
- The deployed dashboard uses the already-built manual index in `ai_output/manual_assistant_index.json`.
- If the training data changes, rebuild locally, commit the regenerated files, and push again.
