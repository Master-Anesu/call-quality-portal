"""
Call Quality Review Portal — Trilogy Care
Flask app that automates call quality reviews for sales reps.
Uses background jobs + polling instead of SSE to avoid Render request timeouts.
"""

import os
import sys
import json
import re
import uuid
import base64
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.brain', 'common_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.brain', 'templates', 'branded', 'base'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['OUTPUT_DIR'] = os.path.join(os.path.dirname(__file__), 'output')

# ---------------------------------------------------------------------------
# In-memory job store  (single-worker gunicorn, so this is safe)
# ---------------------------------------------------------------------------
jobs = {}

# ---------------------------------------------------------------------------
# Config from env vars
# ---------------------------------------------------------------------------
MCP_BASE_URL = "https://llm-alb.trilogycare.com.au/mcp/tools"
MCP_API_KEY = os.environ.get("MCP_API_KEY", "")
WORKSPACE_USER_ID = os.environ.get("WORKSPACE_USER_ID", "")
SENDER_EMAIL = "anesut@trilogycare.com.au"

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "trilogy-gpt-4.1")

AZURE_AD_TENANT_ID = os.environ.get("AZURE_AD_TENANT_ID", "")
AZURE_AD_CLIENT_ID = os.environ.get("AZURE_AD_CLIENT_ID", "")
AZURE_AD_CLIENT_SECRET = os.environ.get("AZURE_AD_CLIENT_SECRET", "")


# ---------------------------------------------------------------------------
# MCP Response Parsers
# ---------------------------------------------------------------------------

def _parse_aircall_calls(result) -> list:
    """Extract calls list from various MCP response shapes."""
    if not result or (isinstance(result, dict) and result.get('error')):
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        if 'calls' in result and isinstance(result['calls'], list):
            return result['calls']
        if 'result' in result:
            inner = result['result']
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict) and 'calls' in inner:
                return inner['calls']
        if 'content' in result:
            content = result['content']
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        try:
                            parsed = json.loads(block['text'])
                            if isinstance(parsed, list):
                                return parsed
                            if isinstance(parsed, dict):
                                if 'calls' in parsed:
                                    return parsed['calls']
                                if 'result' in parsed and isinstance(parsed['result'], dict):
                                    return parsed['result'].get('calls', [])
                        except (json.JSONDecodeError, TypeError):
                            pass
            elif isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict) and 'calls' in parsed:
                        return parsed['calls']
                except (json.JSONDecodeError, TypeError):
                    pass
        if 'raw' in result and isinstance(result['raw'], str):
            try:
                parsed = json.loads(result['raw'])
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict) and 'calls' in parsed:
                    return parsed['calls']
            except (json.JSONDecodeError, TypeError):
                pass
    return []


# ---------------------------------------------------------------------------
# MCP Tool Caller
# ---------------------------------------------------------------------------

def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool via HTTP."""
    import requests as req_lib
    url = f"{MCP_BASE_URL}/{tool_name}/invoke"
    headers = {
        "X-API-Key": MCP_API_KEY,
        "X-User-ID": WORKSPACE_USER_ID,
        "Content-Type": "application/json",
    }
    try:
        resp = req_lib.post(url, json={"arguments": arguments}, headers=headers, timeout=30)
        resp.raise_for_status()
        if not resp.text.strip():
            return {"success": True}
        try:
            return resp.json()
        except ValueError:
            # HTTP 2xx but non-JSON body — the operation succeeded
            return {"success": True, "raw": resp.text[:500]}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Azure OpenAI  (non-streaming — background thread handles the wait)
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
    """Call Azure OpenAI GPT-4.1 (non-streaming). Returns the full response text."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        stream=False,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """You are a call quality reviewer for Trilogy Care, an Australian aged care provider.
You score sales/onboarding calls on 8 dimensions. This is a coaching tool, not a performance audit.
Be specific — ground every score and comment in a real moment from the transcript.

Scoring benchmark: top-performer Aprocina Anthony's conversational style.
Core question for every dimension: "Did this part of the conversation make the caller feel closer to saying yes?"

Return your response as valid JSON with this exact structure:
{
  "client_name": "the client/caller's full name as identified from the transcript",
  "call_story": "2-3 sentence narrative of who called, why, what happened, how it ended",
  "outcome": "brief outcome description",
  "highlights": ["5-8 specific moments where the rep's conversation skills made a difference"],
  "growth_opportunities": [
    {"moment": "what happened", "opportunity": "what could be different", "why_it_matters": "impact on caller experience"}
  ],
  "scores": {
    "rapport": {"score": 0, "comment": "specific feedback"},
    "opening": {"score": 0, "comment": "specific feedback"},
    "reading_the_room": {"score": 0, "comment": "specific feedback"},
    "discovery": {"score": 0, "comment": "specific feedback"},
    "making_value_real": {"score": 0, "comment": "specific feedback"},
    "navigating_resistance": {"score": 0, "comment": "specific feedback"},
    "guiding_to_action": {"score": 0, "comment": "specific feedback"},
    "confidence_knowledge": {"score": 0, "comment": "specific feedback"}
  },
  "top_strength": "the one thing this rep should keep doing",
  "top_development_area": "the one thing that would make the biggest difference",
  "coaching_recommendations": ["2-3 specific, actionable suggestions"]
}

Scoring scales:
- Rapport & Human Connection: 1-10 (weighted 2x, so max 20 points)
- All other dimensions: 1-10 (max 10 points each)
Total max: 90 points

Grade scale:
A+ = 90-100%, A = 80-89%, B = 70-79%, C = 60-69%, D = below 60%
"""


# ---------------------------------------------------------------------------
# Word Document Generator
# ---------------------------------------------------------------------------

def generate_word_doc(review_data: dict) -> str:
    """Generate a branded TC Word document from review data. Returns file path."""
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    NAVY = RGBColor(0x2C, 0x4C, 0x79)
    TEAL = RGBColor(0x43, 0xC0, 0xBE)
    DARK_TEAL = RGBColor(0x00, 0x7F, 0x7E)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    BLACK = RGBColor(0x33, 0x33, 0x33)

    doc = Document()

    # Cover page
    for _ in range(4):
        doc.add_paragraph('')
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run('Call Quality Review')
    title_run.font.size = Pt(36)
    title_run.font.color.rgb = NAVY
    title_run.font.name = 'Aptos'
    title_run.bold = True

    subtitle_para = doc.add_paragraph()
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle_para.add_run(f"{review_data['rep_name']} — {review_data['rep_role']}")
    sub_run.font.size = Pt(18)
    sub_run.font.color.rgb = DARK_TEAL
    sub_run.font.name = 'Aptos'

    tagline_para = doc.add_paragraph()
    tagline_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tag_run = tagline_para.add_run('Conversation Deep-Dive')
    tag_run.font.size = Pt(14)
    tag_run.font.color.rgb = TEAL
    tag_run.font.name = 'Aptos'
    tag_run.italic = True

    doc.add_page_break()

    # Header block
    today_str = datetime.now().strftime('%d %B %Y')
    header_data = [
        ['Agent', f"{review_data['rep_name']} — {review_data.get('est_id', '')} {review_data['rep_role']}"],
        ['Department', 'CSR'],
        ['Manager', 'Anesu Taderera'],
        ['Review Date', today_str],
        ['Call Reviewed', f"{review_data.get('call_date', '')} — {review_data.get('client_name', 'Unknown')} ({review_data.get('direction', '')}, {review_data.get('duration', '')} min)"],
    ]

    table = doc.add_table(rows=len(header_data), cols=2)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (label, value) in enumerate(header_data):
        cell_label = table.cell(i, 0)
        cell_value = table.cell(i, 1)
        cell_label.text = ''
        cell_value.text = ''
        run_l = cell_label.paragraphs[0].add_run(label)
        run_l.bold = True
        run_l.font.size = Pt(11)
        run_l.font.color.rgb = NAVY
        run_l.font.name = 'Aptos'
        run_v = cell_value.paragraphs[0].add_run(value)
        run_v.font.size = Pt(11)
        run_v.font.name = 'Aptos'
        run_v.font.color.rgb = BLACK
        shading = cell_label._element.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:val'): 'clear', qn('w:color'): 'auto', qn('w:fill'): 'E8EDF3',
        })
        shading.append(shading_elem)

    doc.add_paragraph('')

    # Call overview with score
    scores = review_data.get('scores', {})
    total_points = scores.get('rapport', {}).get('score', 0) * 2
    for dim in ['opening', 'reading_the_room', 'discovery', 'making_value_real', 'navigating_resistance', 'guiding_to_action', 'confidence_knowledge']:
        total_points += scores.get(dim, {}).get('score', 0)
    percentage = round((total_points / 90) * 100, 1) if total_points else 0
    overall_score = round(total_points / 9, 1)

    h1 = doc.add_heading(f"{review_data.get('client_name', 'Unknown')} — {review_data.get('call_date', '')} — Score: {overall_score}/10", level=1)
    for run in h1.runs:
        run.font.color.rgb = NAVY

    meta_line = f"{review_data.get('direction', '')} | {review_data.get('duration', '')} minutes | Outcome: {review_data.get('outcome', 'N/A')}"
    meta_para = doc.add_paragraph(meta_line)
    for run in meta_para.runs:
        run.font.size = Pt(10)
        run.font.color.rgb = DARK_TEAL
        run.font.name = 'Aptos'

    # The Story
    h2 = doc.add_heading('The Story', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY
    story_para = doc.add_paragraph(review_data.get('call_story', ''))
    for run in story_para.runs:
        run.font.name = 'Aptos'
        run.font.size = Pt(11)

    # Highlights
    h2 = doc.add_heading('What Worked — Conversation Highlights', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY
    for highlight in review_data.get('highlights', []):
        p = doc.add_paragraph(highlight, style='List Bullet')
        for run in p.runs:
            run.font.name = 'Aptos'
            run.font.size = Pt(11)

    # Growth opportunities
    h2 = doc.add_heading('Where to Level Up — Growth Opportunities', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY
    for opp in review_data.get('growth_opportunities', []):
        if isinstance(opp, dict):
            p = doc.add_paragraph('', style='List Bullet')
            run_moment = p.add_run('The moment: ')
            run_moment.bold = True
            run_moment.font.name = 'Aptos'
            run_moment.font.size = Pt(11)
            run_moment.font.color.rgb = NAVY
            run_text = p.add_run(opp.get('moment', ''))
            run_text.font.name = 'Aptos'
            run_text.font.size = Pt(11)

            p2 = doc.add_paragraph('', style='List Bullet 2')
            run_opp = p2.add_run('The opportunity: ')
            run_opp.bold = True
            run_opp.font.name = 'Aptos'
            run_opp.font.size = Pt(11)
            run_opp.font.color.rgb = TEAL
            run_text2 = p2.add_run(opp.get('opportunity', ''))
            run_text2.font.name = 'Aptos'
            run_text2.font.size = Pt(11)

            p3 = doc.add_paragraph('', style='List Bullet 2')
            run_why = p3.add_run('Why it matters: ')
            run_why.bold = True
            run_why.font.name = 'Aptos'
            run_why.font.size = Pt(11)
            run_text3 = p3.add_run(opp.get('why_it_matters', ''))
            run_text3.font.name = 'Aptos'
            run_text3.font.size = Pt(11)
        else:
            doc.add_paragraph(str(opp), style='List Bullet')

    # Score table
    h2 = doc.add_heading('Overall Assessment', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY

    dim_labels = {
        'rapport': ('Rapport & Human Connection', '2x'),
        'opening': ('Opening & Setting the Scene', '1x'),
        'reading_the_room': ('Reading the Room', '1x'),
        'discovery': ('Conversational Discovery', '1x'),
        'making_value_real': ('Making Value Real', '1x'),
        'navigating_resistance': ('Navigating Resistance', '1x'),
        'guiding_to_action': ('Guiding to Action', '1x'),
        'confidence_knowledge': ('Confidence & Knowledge', '1x'),
    }

    score_table = doc.add_table(rows=len(dim_labels) + 2, cols=4)
    score_table.style = 'Table Grid'
    score_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ['#', 'Dimension', 'Score', 'Weighted']
    for j, h_text in enumerate(headers):
        cell = score_table.cell(0, j)
        cell.text = ''
        run = cell.paragraphs[0].add_run(h_text)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = WHITE
        run.font.name = 'Aptos'
        shading = cell._element.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:val'): 'clear', qn('w:color'): 'auto', qn('w:fill'): '2C4C79',
        })
        shading.append(shading_elem)

    total_weighted = 0
    for i, (key, (label, weight)) in enumerate(dim_labels.items(), start=1):
        score_val = scores.get(key, {}).get('score', 0)
        multiplier = 2 if key == 'rapport' else 1
        weighted = score_val * multiplier
        total_weighted += weighted

        row = score_table.row_cells(i)
        values = [str(i), label, f"{score_val}/10", f"{weighted}/{10 * multiplier}"]
        for j, val in enumerate(values):
            row[j].text = ''
            run = row[j].paragraphs[0].add_run(val)
            run.font.size = Pt(10)
            run.font.name = 'Aptos'
            if i % 2 == 0:
                shading = row[j]._element.get_or_add_tcPr()
                shading_elem = shading.makeelement(qn('w:shd'), {
                    qn('w:val'): 'clear', qn('w:color'): 'auto', qn('w:fill'): 'F0F5F5',
                })
                shading.append(shading_elem)

    # Total row
    total_row = score_table.row_cells(len(dim_labels) + 1)
    total_pct = round((total_weighted / 90) * 100, 1)
    if total_pct >= 90:
        grade = 'A+'
    elif total_pct >= 80:
        grade = 'A'
    elif total_pct >= 70:
        grade = 'B'
    elif total_pct >= 60:
        grade = 'C'
    else:
        grade = 'D'

    total_values = ['', 'TOTAL', f"{total_weighted}/90", f"{total_pct}% ({grade})"]
    for j, val in enumerate(total_values):
        total_row[j].text = ''
        run = total_row[j].paragraphs[0].add_run(val)
        run.bold = True
        run.font.size = Pt(11)
        run.font.name = 'Aptos'
        run.font.color.rgb = NAVY
        shading = total_row[j]._element.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:val'): 'clear', qn('w:color'): 'auto', qn('w:fill'): 'D4E6E6',
        })
        shading.append(shading_elem)

    doc.add_paragraph('')

    # Top strength
    h3 = doc.add_heading('Top Strength', level=3)
    for run in h3.runs:
        run.font.color.rgb = DARK_TEAL
    p = doc.add_paragraph(review_data.get('top_strength', ''))
    for run in p.runs:
        run.font.name = 'Aptos'

    # Top development area
    h3 = doc.add_heading('Top Development Area', level=3)
    for run in h3.runs:
        run.font.color.rgb = NAVY
    p = doc.add_paragraph(review_data.get('top_development_area', ''))
    for run in p.runs:
        run.font.name = 'Aptos'

    # Coaching recommendations
    h3 = doc.add_heading('Coaching Recommendations', level=3)
    for run in h3.runs:
        run.font.color.rgb = NAVY
    for rec in review_data.get('coaching_recommendations', []):
        p = doc.add_paragraph(rec, style='List Bullet')
        for run in p.runs:
            run.font.name = 'Aptos'

    # Footer
    doc.add_paragraph('')
    doc.add_paragraph('_' * 60)
    footer_items = [
        ('Prepared by:', 'Trilogy Care Assistant (AI-generated review)'),
        ('Reviewed by:', 'Anesu Taderera'),
        ('Date:', today_str),
    ]
    for label, value in footer_items:
        p = doc.add_paragraph('')
        run_l = p.add_run(f"{label} ")
        run_l.bold = True
        run_l.font.size = Pt(9)
        run_l.font.name = 'Aptos'
        run_l.font.color.rgb = NAVY
        run_v = p.add_run(value)
        run_v.font.size = Pt(9)
        run_v.font.name = 'Aptos'

    # Save
    first = review_data['rep_name'].split()[0] if review_data['rep_name'] else 'Unknown'
    last = review_data['rep_name'].split()[-1] if len(review_data['rep_name'].split()) > 1 else ''
    month_year = datetime.now().strftime('%b%Y')
    filename = f"{first}_{last}_Call_Quality_Review_{month_year}.docx"
    filepath = os.path.join(app.config['OUTPUT_DIR'], filename)
    os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
    doc.save(filepath)

    review_data['filename'] = filename
    review_data['filepath'] = filepath
    review_data['grade'] = grade
    review_data['total_points'] = total_weighted
    review_data['percentage'] = total_pct

    return filepath


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check — useful for verifying the app started correctly."""
    return jsonify({'status': 'ok', 'jobs_store': len(jobs)})


@app.route('/api/search-employee', methods=['POST'])
def search_employee():
    """Search for an employee by name."""
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    result = call_mcp_tool('search_employees', {'query': name})
    return jsonify(result)


@app.route('/api/aircall-users', methods=['GET'])
def get_aircall_users():
    """Get Aircall users list."""
    result = call_mcp_tool('list_aircall_users', {})
    return jsonify(result)


@app.route('/api/calls', methods=['POST'])
def get_calls():
    """Get recent calls for an Aircall user by email with date filter support."""
    user_email = request.json.get('user_email', '').strip()
    date_filter = request.json.get('date', '').strip()
    if not user_email:
        return jsonify({'error': 'user_email is required'}), 400

    if date_filter == 'today' or not date_filter:
        date_from = datetime.now().strftime('%Y-%m-%d')
    elif date_filter == 'week':
        date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    else:
        date_from = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    result = call_mcp_tool('list_aircall_calls', {
        'user_email': user_email,
        'limit': 200,
        'date_from': date_from,
    })

    all_calls = _parse_aircall_calls(result)

    # Only include calls over 5 minutes (300 seconds)
    calls = []
    for c in all_calls:
        dur = c.get('duration') or 0
        if isinstance(dur, str):
            try:
                dur = int(dur)
            except ValueError:
                dur = 0
        if dur < 300:
            continue
        # Normalize recording fields so frontend can detect them
        if c.get('recording') and not c.get('recording_url'):
            c['recording_url'] = c['recording']
        if c.get('recording') or c.get('recording_url'):
            c['has_recording'] = True
        calls.append(c)

    today_str = datetime.now().strftime('%Y-%m-%d')
    unique_clients = set()
    today_calls = 0
    for c in calls:
        call_date = c.get('created_at', '')[:10]
        if call_date == today_str:
            today_calls += 1
            client_id = c.get('contact_name') or c.get('raw_digits') or c.get('phone_number') or ''
            if client_id:
                unique_clients.add(client_id.lower())

    return jsonify({
        'result': {
            'calls': calls,
            'stats': {
                'total_calls': len(calls),
                'today_calls': today_calls,
                'unique_clients_today': len(unique_clients),
                'date_from': date_from,
            }
        }
    })


@app.route('/api/start-review', methods=['POST'])
def start_review():
    """Kick off the review pipeline in a background thread. Returns a job_id for polling."""
    data = request.json
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        'status': 'running',
        'message': 'Fetching call transcript...',
        'result': None,
        'error': None,
    }

    thread = threading.Thread(target=run_review_pipeline, args=(job_id, data), daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/api/review-status/<job_id>', methods=['GET'])
def review_status(job_id):
    """Poll for the status of a review job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)



@app.route('/api/start-review-from-transcript', methods=['POST'])
def start_review_from_transcript():
    """Start a review from an uploaded transcript file. Extracts rep name from content."""
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'error': 'Please upload a transcript file.'}), 400
    try:
        raw = file.read()
        transcript = raw.decode('utf-8', errors='replace').strip()
    except Exception as e:
        return jsonify({'error': f'Could not read file: {str(e)}'}), 400
    if len(transcript) < 100:
        return jsonify({'error': 'Transcript seems too short. Please upload the full call transcript.'}), 400

    rep_name = ''
    m = re.search(r'(?:agent|rep|representative|advisor)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', transcript[:2000], re.IGNORECASE)
    if m: rep_name = m.group(1).strip()
    if not rep_name:
        m = re.search(r'\[agent\]\s*([A-Z][a-z]+ ?[A-Z]?[a-z]*)', transcript[:2000])
        if m: rep_name = m.group(1).strip()
    if not rep_name:
        m = re.search(r"(?:my name is|this is|I'm|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:from|at|with)\s+Trilogy", transcript[:3000], re.IGNORECASE)
        if m: rep_name = m.group(1).strip()
    if not rep_name:
        base = os.path.splitext(file.filename)[0]
        parts = re.split(r'[_\-\s]+', base)
        name_parts = [p for p in parts if p[0:1].isupper() and len(p) > 1 and p.lower() not in ('call', 'transcript', 'review', 'recording', 'aircall')]
        if len(name_parts) >= 2: rep_name = ' '.join(name_parts[:2])
    if not rep_name:
        return jsonify({'error': 'Could not detect the rep name from the transcript. Please rename the file to include the rep name (e.g. "Sarah Palmer transcript.txt").'}), 400

    rep_email = ''
    rep_role = ''
    est_id = ''
    emp_result = call_mcp_tool('search_employees', {'query': rep_name})
    if isinstance(emp_result, dict) and not emp_result.get('error'):
        employees = []
        inner = emp_result.get('result', emp_result)
        if isinstance(inner, dict): employees = inner.get('employees', inner.get('results', []))
        elif isinstance(inner, list): employees = inner
        for emp in (employees if isinstance(employees, list) else []):
            emp_name = emp.get('full_name', '') or emp.get('name', '')
            if emp_name and rep_name.lower() in emp_name.lower():
                title_raw = emp.get('title', '')
                est_match = re.match(r'(EST-\d+)\s*(.*)', title_raw)
                est_id = emp.get('employee_id', '') or emp.get('est_id', '') or (est_match.group(1) if est_match else '')
                rep_role = emp.get('role', '') or emp.get('job_title', '') or (est_match.group(2) if est_match else title_raw)
                rep_email = emp.get('email', '') or emp.get('work_email', '')
                fn = emp.get('full_name', '') or emp.get('name', '')
                if fn: rep_name = fn
                break

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {'status': 'running', 'message': 'AI is scoring the call...', 'result': None, 'error': None}
    pipeline_data = {
        'rep_name': rep_name, 'rep_email': rep_email,
        'rep_role': rep_role, 'est_id': est_id,
        'call_date': datetime.now().strftime('%d %b %Y'),
        'call_direction': '', 'call_duration': '',
        'client_name': 'Unknown', 'client_phone': '',
        'transcript_text': transcript,
    }
    thread = threading.Thread(target=run_review_pipeline, args=(job_id, pipeline_data), daemon=True)
    thread.start()
    return jsonify({'job_id': job_id, 'rep_name': rep_name, 'rep_email': rep_email, 'rep_role': rep_role, 'est_id': est_id})


def run_review_pipeline(job_id: str, data: dict):
    """Background worker: fetch transcript, score with LLM, generate doc."""
    try:
        call_id = data.get('call_id')
        rep_name = data.get('rep_name', '')
        rep_email = data.get('rep_email', '')
        rep_role = data.get('rep_role', '')
        est_id = data.get('est_id', '')
        call_date = data.get('call_date', '')
        call_direction = data.get('call_direction', '')
        call_duration = data.get('call_duration', '')
        client_name = data.get('client_name', 'Unknown')
        client_phone = data.get('client_phone', '')
        direct_link = data.get('direct_link', '')
        recording_url = data.get('recording_url', '')

        # Step 1: Get transcript (pre-provided or fetched from Aircall)
        transcript_text = data.get('transcript_text', '')
        resolved_client = client_name

        def fetch_transcript():
            if data.get('transcript_text'):
                return {'result': {'transcript': data['transcript_text']}}
            result = call_mcp_tool('get_aircall_transcript', {'call_id': str(call_id)})
            # Fallback: if get_aircall_transcript fails, try search_aircall_transcripts
            if isinstance(result, dict):
                inner = result.get('result', result)
                has_segments = isinstance(inner, dict) and inner.get('transcript_segments')
                has_raw = isinstance(inner, dict) and (inner.get('raw') or inner.get('transcript'))
                has_error = result.get('error') or (isinstance(inner, dict) and inner.get('success') is False)
                if (not has_segments and not has_raw) or has_error:
                    import logging
                    logging.getLogger(__name__).info(f'get_aircall_transcript returned no data for {call_id}, trying search_aircall_transcripts...')
                    search_result = call_mcp_tool('search_aircall_transcripts', {'call_id': str(call_id)})
                    if isinstance(search_result, dict) and not search_result.get('error'):
                        return search_result
            return result

        def fetch_client():
            if client_name != 'Unknown' or not client_phone:
                return None
            return call_mcp_tool('search_customers', {'query': client_phone})

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_transcript = pool.submit(fetch_transcript)
            future_client = pool.submit(fetch_client)

            transcript_result = future_transcript.result()
            if isinstance(transcript_result, dict):
                if transcript_result.get('error'):
                    jobs[job_id] = {'status': 'error', 'message': f"Failed to get transcript: {transcript_result['error']}", 'result': None, 'error': transcript_result['error']}
                    return
                result = transcript_result.get('result', transcript_result)
                segments = result.get('transcript_segments', [])
                if segments:
                    transcript_text = '\n'.join(
                        f"[{seg.get('participant_type', 'unknown')}]: {seg.get('text', '')}"
                        for seg in segments if seg.get('text')
                    )
                else:
                    transcript_text = result.get('raw', '') or result.get('transcript', '') or ''
            else:
                transcript_text = str(transcript_result)

            if not transcript_text and recording_url:
                jobs[job_id]['message'] = 'No stored transcript — transcribing from recording (live)...'
                logger.info('No transcript in API for call %s, transcribing recording live', call_id)
                transcript_text = transcribe_recording(recording_url)

            if not transcript_text:
                jobs[job_id] = {'status': 'error', 'message': 'No transcript or recording available for this call.', 'result': None, 'error': 'No transcript'}
                return

            client_result = future_client.result()
            if client_result and isinstance(client_result, dict) and not client_result.get('error'):
                raw = client_result.get('raw', '') or json.dumps(client_result)
                resolved_client = extract_client_name(raw) or client_name

        # Step 2: Score with LLM
        jobs[job_id]['message'] = 'AI is scoring the call across 8 dimensions...'

        scoring_prompt = f"""Score this call quality review.

REP (the Trilogy Care employee being reviewed): {rep_name} ({rep_role})
CLIENT (the external caller/recipient): {resolved_client}
CALL DATE: {call_date}
DIRECTION: {call_direction}
DURATION: {call_duration} minutes

IMPORTANT: In the transcript below, [agent] is ALWAYS {rep_name} (the rep being reviewed). [external] is ALWAYS the client/caller. Do NOT confuse these roles. When writing your review, refer to {rep_name} as the rep and {resolved_client} as the client.

TRANSCRIPT:
{transcript_text[:12000]}

Score each dimension 1-10 based on the rubric. Be specific and reference real moments from the transcript. Return valid JSON only."""

        try:
            llm_response = call_llm(SCORING_SYSTEM_PROMPT, scoring_prompt, max_tokens=4000)
            review_json = parse_json_from_response(llm_response)
        except Exception as e:
            jobs[job_id] = {'status': 'error', 'message': f"LLM scoring failed: {str(e)}", 'result': None, 'error': str(e)}
            return

        if not review_json or not review_json.get('scores'):
            jobs[job_id] = {'status': 'error', 'message': 'LLM returned invalid or incomplete JSON. Please retry.', 'result': None, 'error': 'Invalid JSON from LLM'}
            return

        # Use LLM-identified client name if still Unknown
        llm_client = review_json.get('client_name', '')
        if (not resolved_client or resolved_client == 'Unknown') and llm_client:
            resolved_client = llm_client

        # Step 3: Generate Word doc
        jobs[job_id]['message'] = 'Generating Word document...'

        review_data = {
            'rep_name': rep_name,
            'rep_email': rep_email,
            'rep_role': rep_role,
            'est_id': est_id,
            'call_id': call_id,
            'call_date': call_date,
            'direction': call_direction,
            'duration': call_duration,
            'client_name': resolved_client,
            'client_phone': client_phone,
            'call_story': review_json.get('call_story', ''),
            'outcome': review_json.get('outcome', ''),
            'highlights': review_json.get('highlights', []),
            'growth_opportunities': review_json.get('growth_opportunities', []),
            'scores': review_json.get('scores', {}),
            'top_strength': review_json.get('top_strength', ''),
            'top_development_area': review_json.get('top_development_area', ''),
            'coaching_recommendations': review_json.get('coaching_recommendations', []),
        }

        try:
            filepath = generate_word_doc(review_data)
        except Exception as e:
            jobs[job_id] = {'status': 'error', 'message': f"Document generation failed: {str(e)}", 'result': None, 'error': str(e)}
            return

        call_link = f"https://assets.aircall.io/calls/{call_id}/recording"
        review_data['call_link'] = call_link
        review_data['filepath'] = filepath
        review_data['filename'] = os.path.basename(filepath)

        # Document ready — attachment will be sent inline via MS Graph at email time

        jobs[job_id] = {'status': 'complete', 'message': 'Review complete!', 'result': review_data, 'error': None}

    except Exception as e:
        jobs[job_id] = {'status': 'error', 'message': f"Unexpected error: {str(e)}", 'result': None, 'error': str(e)}


@app.route('/api/send-email', methods=['POST'])
def send_email_route():
    """Send the review document via email with .docx attachment using Microsoft Graph."""
    import requests as req_lib
    data = request.json
    rep_email = data.get('rep_email', '')
    rep_name = data.get('rep_name', '')
    first_name = rep_name.split()[0] if rep_name else ''
    client_name = data.get('client_name', '')
    call_date = data.get('call_date', '')
    call_link = data.get('call_link', '')
    filepath = data.get('filepath', '')
    filename = data.get('filename', '')

    if not rep_email:
        return jsonify({'error': 'No recipient email address provided.'}), 400

    subject = f"Call Quality Review \u2014 {client_name} \u2014 {call_date}"
    body_text = f"""Hi {first_name},

Please find attached your call quality review for your call with {client_name} on {call_date}:
{call_link}

Have a read through and let me know if you'd like to chat about anything in the review.

Kind regards,
Anesu"""

    # Build Microsoft Graph sendMail payload with inline attachment
    mail_payload = {
        'message': {
            'subject': subject,
            'body': {'contentType': 'Text', 'content': body_text},
            'from': {'emailAddress': {'address': SENDER_EMAIL}},
            'toRecipients': [{'emailAddress': {'address': rep_email}}],
        },
        'saveToSentItems': True,
    }

    # Attach the Word document if available on disk
    attachment_ok = False
    if filepath and os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
            mail_payload['message']['attachments'] = [{
                '@odata.type': '#microsoft.graph.fileAttachment',
                'name': filename or os.path.basename(filepath),
                'contentBytes': base64.b64encode(file_bytes).decode('utf-8'),
                'contentType': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            }]
            attachment_ok = True
            logger.info('Attached %s (%d bytes)', filename, len(file_bytes))
        except Exception as e:
            logger.error('Failed to read file for attachment: %s', e)

    try:
        # Get Azure AD token
        token = get_graph_token()
        if not token:
            return jsonify({'error': 'Failed to get Microsoft Graph token'}), 500

        resp = req_lib.post(
            f'https://graph.microsoft.com/v1.0/users/{SENDER_EMAIL}/sendMail',
            json=mail_payload,
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            timeout=30,
        )
        logger.info('Graph sendMail status: %s', resp.status_code)
        if resp.status_code == 202:
            return jsonify({'success': True, 'attachment_included': attachment_ok})
        else:
            error_text = resp.text[:500]
            logger.error('Graph sendMail failed: %s %s', resp.status_code, error_text)
            return jsonify({'error': f'Microsoft Graph error: {resp.status_code}'}), 500
    except Exception as e:
        logger.error('send_email failed: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/output/<filename>')
def download_file(filename):
    """Download a generated document."""
    return send_from_directory(app.config['OUTPUT_DIR'], filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_graph_token():
    """Get a Microsoft Graph API access token via Azure AD client credentials."""
    import requests as req_lib
    if not all([AZURE_AD_TENANT_ID, AZURE_AD_CLIENT_ID, AZURE_AD_CLIENT_SECRET]):
        logger.error('Azure AD credentials not configured')
        return None
    try:
        resp = req_lib.post(
            f'https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/oauth2/v2.0/token',
            data={
                'grant_type': 'client_credentials',
                'client_id': AZURE_AD_CLIENT_ID,
                'client_secret': AZURE_AD_CLIENT_SECRET,
                'scope': 'https://graph.microsoft.com/.default',
            },
            timeout=10,
        )
        data = resp.json()
        return data.get('access_token')
    except Exception as e:
        logger.error('Failed to get Graph token: %s', e)
        return None


def extract_client_name(raw: str) -> str:
    """Try to extract a client name from search results (various MCP response shapes)."""
    try:
        data = json.loads(raw)
        # Unwrap nested MCP response: {result: {customers: [...]}} or {result: [...]}
        if isinstance(data, dict):
            inner = data.get('result', data)
            if isinstance(inner, dict):
                # {result: {customers: [...]}} or {customers: [...]}
                for key in ('customers', 'contacts', 'results', 'data'):
                    if isinstance(inner.get(key), list) and inner[key]:
                        inner = inner[key]
                        break
            if isinstance(inner, list) and inner:
                item = inner[0]
                return (item.get('display_name', '') or item.get('full_name', '')
                        or item.get('name', '')
                        or f"{item.get('first_name', '')} {item.get('last_name', '')}".strip())
            if isinstance(inner, dict):
                return (inner.get('display_name', '') or inner.get('full_name', '')
                        or inner.get('name', '')
                        or f"{inner.get('first_name', '')} {inner.get('last_name', '')}".strip())
        if isinstance(data, list) and data:
            item = data[0]
            return (item.get('display_name', '') or item.get('full_name', '')
                    or item.get('name', '')
                    or f"{item.get('first_name', '')} {item.get('last_name', '')}".strip())
    except Exception:
        pass
    return ''


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response (may be wrapped in markdown code blocks)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
