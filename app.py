"""
Call Quality Review Portal — Trilogy Care
Flask app that automates call quality reviews for sales reps.
"""

import os
import sys
import json
import subprocess
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, request, Response

# Add common functions to path for TC doc helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.brain', 'common_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.brain', 'templates', 'branded', 'base'))

app = Flask(__name__)
app.config['OUTPUT_DIR'] = os.path.join(os.path.dirname(__file__), 'output')

MCP_BASE_URL = "https://llm-alb.trilogycare.com.au/mcp/tools"
MCP_API_KEY = os.environ.get("MCP_API_KEY", "")
WORKSPACE_USER_ID = os.environ.get("WORKSPACE_USER_ID", "")
SENDER_EMAIL = "anesut@trilogycare.com.au"

# Azure OpenAI config
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "trilogy-gpt-4.1")


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
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
    """Call Azure OpenAI GPT-4.1 for scoring and review generation."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Scoring prompt (from call-quality-review-template.md)
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """You are a call quality reviewer for Trilogy Care, an Australian aged care provider.
You score sales/onboarding calls on 8 dimensions. This is a coaching tool, not a performance audit.
Be specific — ground every score and comment in a real moment from the transcript.

Scoring benchmark: top-performer Aprocina Anthony's conversational style.
Core question for every dimension: "Did this part of the conversation make the caller feel closer to saying yes?"

Return your response as valid JSON with this exact structure:
{
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
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    # Colors
    NAVY = RGBColor(0x2C, 0x4C, 0x79)
    TEAL = RGBColor(0x43, 0xC0, 0xBE)
    DARK_TEAL = RGBColor(0x00, 0x7F, 0x7E)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    RED = RGBColor(0xE0, 0x4B, 0x51)
    BLACK = RGBColor(0x33, 0x33, 0x33)

    doc = Document()

    # --- Cover page ---
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

    # --- Header Block ---
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
        # Navy background for label column
        shading = cell_label._element.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:val'): 'clear',
            qn('w:color'): 'auto',
            qn('w:fill'): 'E8EDF3',
        })
        shading.append(shading_elem)

    doc.add_paragraph('')

    # --- Section 1: Call Overview ---
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

    # --- Section 2: What Worked ---
    h2 = doc.add_heading('What Worked — Conversation Highlights', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY

    for highlight in review_data.get('highlights', []):
        p = doc.add_paragraph(highlight, style='List Bullet')
        for run in p.runs:
            run.font.name = 'Aptos'
            run.font.size = Pt(11)

    # --- Section 3: Where to Level Up ---
    h2 = doc.add_heading('Where to Level Up — Growth Opportunities', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY

    for opp in review_data.get('growth_opportunities', []):
        if isinstance(opp, dict):
            p = doc.add_paragraph('', style='List Bullet')
            run_moment = p.add_run(f"The moment: ")
            run_moment.bold = True
            run_moment.font.name = 'Aptos'
            run_moment.font.size = Pt(11)
            run_moment.font.color.rgb = NAVY
            run_text = p.add_run(opp.get('moment', ''))
            run_text.font.name = 'Aptos'
            run_text.font.size = Pt(11)

            p2 = doc.add_paragraph('', style='List Bullet 2')
            run_opp = p2.add_run(f"The opportunity: ")
            run_opp.bold = True
            run_opp.font.name = 'Aptos'
            run_opp.font.size = Pt(11)
            run_opp.font.color.rgb = TEAL
            run_text2 = p2.add_run(opp.get('opportunity', ''))
            run_text2.font.name = 'Aptos'
            run_text2.font.size = Pt(11)

            p3 = doc.add_paragraph('', style='List Bullet 2')
            run_why = p3.add_run(f"Why it matters: ")
            run_why.bold = True
            run_why.font.name = 'Aptos'
            run_why.font.size = Pt(11)
            run_text3 = p3.add_run(opp.get('why_it_matters', ''))
            run_text3.font.name = 'Aptos'
            run_text3.font.size = Pt(11)
        else:
            p = doc.add_paragraph(str(opp), style='List Bullet')

    # --- Section 4: Overall Assessment ---
    h2 = doc.add_heading('Overall Assessment', level=2)
    for run in h2.runs:
        run.font.color.rgb = NAVY

    # Score table
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

    # Header row
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
            # Alternate row shading
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

    # --- Footer ---
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
    """Get recent calls for an Aircall user by email. Pulls 90 days of history."""
    user_email = request.json.get('user_email', '')
    date_from = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    result = call_mcp_tool('list_aircall_calls', {
        'user_email': user_email,
        'limit': 100,
        'date_from': date_from,
    })
    return jsonify(result)


@app.route('/api/generate-review', methods=['POST'])
def generate_review():
    """Full pipeline: get transcript, score, generate doc. Returns SSE stream."""
    data = request.json
    call_id = data.get('call_id')
    rep_name = data.get('rep_name', '')
    rep_email = data.get('rep_email', '')
    rep_role = data.get('rep_role', '')
    est_id = data.get('est_id', '')
    aircall_user_id = data.get('aircall_user_id', '')
    call_date = data.get('call_date', '')
    call_direction = data.get('call_direction', '')
    call_duration = data.get('call_duration', '')
    client_name = data.get('client_name', 'Unknown')
    client_phone = data.get('client_phone', '')

    def stream():
        # Step 1: Get transcript
        yield sse_msg('status', 'Fetching call transcript...')
        transcript_result = call_mcp_tool('get_aircall_transcript', {'call_id': str(call_id)})
        transcript_text = ''
        if isinstance(transcript_result, dict):
            if 'error' in transcript_result:
                yield sse_msg('error', f"Failed to get transcript: {transcript_result['error']}")
                return
            transcript_text = transcript_result.get('raw', '') or transcript_result.get('transcript', '') or json.dumps(transcript_result)
        else:
            transcript_text = str(transcript_result)

        if not transcript_text or transcript_text == '{}':
            yield sse_msg('error', 'No transcript available for this call.')
            return

        yield sse_msg('status', 'Transcript loaded. Scoring call...')

        # Step 2: Try to identify client from phone number
        if client_name == 'Unknown' and client_phone:
            yield sse_msg('status', 'Looking up client...')
            client_result = call_mcp_tool('search_customers', {'query': client_phone})
            if isinstance(client_result, dict) and not client_result.get('error'):
                # Try to extract name from result
                raw = client_result.get('raw', '') or json.dumps(client_result)
                client_name = extract_client_name(raw) or 'Unknown'

        # Step 3: Score with LLM
        yield sse_msg('status', 'AI is scoring the call across 8 dimensions...')

        scoring_prompt = f"""Score this call quality review.

REP: {rep_name} ({rep_role})
CALL DATE: {call_date}
DIRECTION: {call_direction}
DURATION: {call_duration} minutes
CLIENT: {client_name}

TRANSCRIPT:
{transcript_text[:12000]}

Score each dimension 1-10 based on the rubric. Be specific and reference real moments from the transcript. Return valid JSON only."""

        try:
            llm_response = call_llm(SCORING_SYSTEM_PROMPT, scoring_prompt, max_tokens=4000)
            # Parse JSON from response
            review_json = parse_json_from_response(llm_response)
        except Exception as e:
            yield sse_msg('error', f"LLM scoring failed: {str(e)}")
            return

        yield sse_msg('status', 'Generating Word document...')

        # Step 4: Build review data
        review_data = {
            'rep_name': rep_name,
            'rep_email': rep_email,
            'rep_role': rep_role,
            'est_id': est_id,
            'call_id': call_id,
            'call_date': call_date,
            'direction': call_direction,
            'duration': call_duration,
            'client_name': client_name,
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

        # Step 5: Generate Word doc
        try:
            filepath = generate_word_doc(review_data)
            yield sse_msg('status', 'Document generated successfully!')
        except Exception as e:
            yield sse_msg('error', f"Document generation failed: {str(e)}")
            return

        # Step 6: Return review summary
        call_link = f"https://app.aircall.io/calls/{call_id}"
        review_data['call_link'] = call_link
        review_data['filepath'] = filepath
        review_data['filename'] = os.path.basename(filepath)

        yield sse_msg('complete', json.dumps(review_data))

    return Response(stream(), mimetype='text/event-stream')


@app.route('/api/send-email', methods=['POST'])
def send_email():
    """Send the review document via email."""
    data = request.json
    rep_email = data.get('rep_email', '')
    rep_name = data.get('rep_name', '')
    first_name = rep_name.split()[0] if rep_name else ''
    client_name = data.get('client_name', '')
    call_date = data.get('call_date', '')
    call_link = data.get('call_link', '')
    filename = data.get('filename', '')
    filepath = data.get('filepath', '')
    month_year = datetime.now().strftime('%B %Y')

    subject = f"Call Quality Review — {month_year}"
    body = f"""Hi {first_name},

Please find attached your call quality review for this month.

The review is based on your call with {client_name} on {call_date}:
{call_link}

Have a read through and let me know if you'd like to chat about anything in the review.

Kind regards,
Anesu"""

    # Upload file to OneDrive first so we can attach it
    # Then send email with attachment
    try:
        result = call_mcp_tool('send_email', {
            'to': rep_email,
            'subject': subject,
            'body': body,
            'from': SENDER_EMAIL,
        })
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/output/<filename>')
def download_file(filename):
    """Download a generated document."""
    from flask import send_from_directory
    return send_from_directory(app.config['OUTPUT_DIR'], filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sse_msg(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def extract_client_name(raw: str) -> str:
    """Try to extract a client name from search results."""
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            return data[0].get('name', '') or data[0].get('full_name', '')
        if isinstance(data, dict):
            return data.get('name', '') or data.get('full_name', '')
    except:
        pass
    return ''


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response (may be wrapped in markdown code blocks)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    # Try extracting from code blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    # Try finding JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
