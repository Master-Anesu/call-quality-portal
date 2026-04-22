# Call Quality Review Portal — Trilogy Care

AI-powered call quality review tool for sales/onboarding representatives.

## What it does

1. **Search** for a sales rep by name
2. **Select** a recent Aircall call to review
3. **AI scores** the call across 8 dimensions (90-point scale)
4. **Generates** a branded Word document with coaching feedback
5. **Emails** the review directly to the rep

## Scoring Dimensions

| # | Dimension | Weight |
|---|-----------|--------|
| 1 | Rapport & Human Connection | 2x |
| 2 | Opening & Setting the Scene | 1x |
| 3 | Reading the Room | 1x |
| 4 | Conversational Discovery | 1x |
| 5 | Making Value Real | 1x |
| 6 | Navigating Resistance | 1x |
| 7 | Guiding to Action | 1x |
| 8 | Confidence & Knowledge | 1x |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MCP_API_KEY` | Trilogy Care MCP service API key |
| `WORKSPACE_USER_ID` | MCP workspace user identifier |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2025-01-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name (default: `trilogy-gpt-4.1`) |

## Local Development

```bash
pip install -r requirements.txt
export MCP_API_KEY=your_key
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
python app.py
```

Open http://localhost:5050

## Deployment

Configured for Render.com — see `render.yaml`. Connect the GitHub repo and set the environment variables in the Render dashboard.
