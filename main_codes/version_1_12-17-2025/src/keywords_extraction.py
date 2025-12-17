import os
import json
import asyncio
from typing import Any, Dict, List, Tuple
from html import unescape
import re
import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import requests

# Ensure project root is on sys.path so 'configs' and other top-level packages can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Uses your existing LLM config
from configs.llm_config import chat_completions, BASIC_AUTH, AUTH_URL

# --- Logging setup ---
LOG_DIR = PROJECT_ROOT / "logs" / "keywords_extraction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_logger = logging.getLogger("keywords_extraction")
_logger.setLevel(logging.INFO)

_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

if not _logger.handlers:
    _logger.addHandler(_file_handler)
    _logger.addHandler(_console_handler)

def log_info(msg: str):
    _logger.info(msg)

def log_error(msg: str, exc: Exception | None = None):
    if exc:
        _logger.error(f"{msg} | Exception: {exc}", exc_info=True)
    else:
        _logger.error(msg)

INPUT_FOLDER = 'inputs/careguides_input_samples'
OUTPUT_FOLDER = "outputs/extracted_keywords_output_samples/"

# --- Auth (optional; kept if your chat_completions needs it) ---
def get_access_token():
    headers = {
        "Authorization": f"Basic {BASIC_AUTH}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    try:
        resp = requests.post(AUTH_URL, headers=headers, data=data, timeout=15)
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            log_info("Access token retrieved")
            return token
        else:
            log_error(f"Failed to get access token: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        log_error("Auth request failed", e)
        return None


# --- Tiny HTML cleaner (no external libs) ---
def html_to_text(html: str) -> str:
    """Convert HTML to plain text: unescape entities, remove tags, normalize spaces."""
    if not isinstance(html, str):
        return ""
    # Unescape entities like &nbsp;, &lt;, &gt;
    text = unescape(html)
    # Replace <br> and block tags with newlines to keep structure
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*p\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*li\s*>", "\n• ", text)
    text = re.sub(r"(?i)</\s*tr\s*>", "\n", text)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --- Prompt builder (compact) ---
def build_prompt(title: str, context: str) -> str:
    return f"""
Return STRICT JSON ONLY. No commentary.

Extract exact domain keywords from the Context Only. Group into:
- medical_terms
- drug_names
- diagnoses
- procedures
- tests
- devices
- organizations
- locations
- person_names
- dates
- numbers

Rules:
- Preserve exact technical terms and names.
- Do not invent facts or add anything
- Do not add, modify, paraphrase, or invent any text.
- Empty arrays if none.

Input:
Title: "{title}"
Context: "{context}"

Output JSON:
{{
  "keywords": {{
    "medical_terms": [],
    "drug_names": [],
    "diagnoses": [],
    "procedures": [],
    "tests": [],
    "devices": [],
    "organizations": [],
    "locations": [],
    "person_names": [],
    "dates": [],
    "numbers": []
  }}
}}
""".strip()


# --- LLM call wrapper ---
async def extract_keywords(title: str, context: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": build_prompt(title, context)}]
    try:
        result = await chat_completions(messages)
        content = result.get("final_result", "")
        # Parse JSON block safely
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])
        return parsed
    except Exception as e:
        log_error(f"LLM JSON parse failed for '{title[:60]}...'", e)
        # Safe empty fallback
        return {
            "keywords": {
                "medical_terms": [],
                "drug_names": [],
                "diagnoses": [],
                "procedures": [],
                "tests": [],
                "devices": [],
                "organizations": [],
                "locations": [],
                "person_names": [],
                "dates": [],
                "numbers": []
            }
        }


# --- Find (title, content/context) pairs; tuned for your sample ---
def find_pairs(data: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    def try_add(obj: Dict[str, Any]):
        title = None
        context = None
        # Common keys
        if isinstance(obj, dict):
            if "title" in obj and isinstance(obj["title"], str):
                title = obj["title"]
            # prefer 'content' if present, else 'context' or 'text'
            for ck in ("content", "context", "text", "body", "description"):
                if ck in obj and isinstance(obj[ck], str) and obj[ck].strip():
                    context = obj[ck]
                    break
        if title and context:
            pairs.append((title, context))

    def walk(node: Any):
        if isinstance(node, dict):
            # Direct add
            try_add(node)
            # Known containers: 'diagnosis' from your sample plus generic names
            for key in ("diagnosis", "items", "records", "data", "list", "entries", "results"):
                if key in node:
                    walk(node[key])
            # Dive deeper
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item)
        # primitives ignored

    walk(data)
    log_info(f"Found {len(pairs)} (title, content) pairs")
    return pairs


# --- Process a single file (sequential) ---
async def process_file(input_path: str) -> Dict[str, Any]:
    log_info(f"Processing file: {os.path.basename(input_path)}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_error(f"Failed reading JSON: {input_path}", e)
        return {"source_file": os.path.basename(input_path), "results": []}

    pairs = find_pairs(data)
    if not pairs:
        log_info(f"No (title, content) pairs in {os.path.basename(input_path)}")
        return {"source_file": os.path.basename(input_path), "results": []}

    results = []
    for title, raw_html in pairs:
        try:
            context_text = html_to_text(raw_html)
            parsed = await extract_keywords(title, context_text)
            results.append({
                "title": title,
                "context": context_text,
                "keywords": parsed.get("keywords", {})
            })
            log_info(f"Extracted keywords for: {title}")
        except Exception as e:
            log_error(f"Extraction failed for: {title}", e)

    return {"source_file": os.path.basename(input_path), "results": results}


# --- Main: iterate input folder and write outputs ---
async def main(input_folder: str, output_folder: str):
    # Optional token check (depends on your chat_completions)
    _ = get_access_token()

    os.makedirs(output_folder, exist_ok=True)

    try:
        files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(".json")
        ]
    except Exception as e:
        log_error("Listing input folder failed", e)
        return

    if not files:
        log_info("No JSON files found.")
        return

    for fpath in files:
        try:
            log_info(f"→ Processing {os.path.basename(fpath)}")
            out_data = await process_file(fpath)
            out_name = os.path.splitext(os.path.basename(fpath))[0] + "_keywords.json"
            out_path = os.path.join(output_folder, out_name)
            with open(out_path, "w", encoding="utf-8") as fo:
                json.dump(out_data, fo, ensure_ascii=False, indent=2)
            log_info(f"Saved: {out_path}")
        except Exception as e:
            log_error(f"Pipeline error for {os.path.basename(fpath)}", e)


if __name__ == "__main__":
    asyncio.run(main(INPUT_FOLDER, OUTPUT_FOLDER))