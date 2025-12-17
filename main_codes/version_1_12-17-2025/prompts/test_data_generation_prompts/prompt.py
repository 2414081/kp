# ...existing code...
import json

def section_prompt(diagnosis_title: str, section_name: str, section_text: str) -> str:
    return f"""
Role: You are an expert in generating clinician-focused medical questions.
Task: Generate concise clinician-facing questions based on the provided diagnosis title and section text.

Rules:
- Preserve the diagnosis_title EXACTLY as provided (do not change words or spellings).
- If section_text is available, use 2 to 3 salient keywords from the section_text EXACTLY as written (preserve spellings/case) within basic_questions, validation_question, and deepdive_question.
- Do NOT include section_name in any question.
- If section_text is empty or not available, use ONLY diagnosis_title (preserve words and spellings) to generate questions.
- The questions must be realistic, clinician-focused, and actionable.
- Do NOT include lists, quotes, code fences, or markdown.
- Return ONLY a valid JSON object with EXACT keys: diagnosis_title, section, items.
- Items must be an array with ONE object having EXACT keys: basic_questions, validation_question, deepdive_question.
- basic_questions must be an array of EXACTLY 3 short clinical questions (each ≤ 18 words). Do not return an empty array.
- deepdive_question must be ONE short deeper clinical question (≤ 22 words) that probes mechanisms, thresholds, decision points, or nuanced application.

Phrasing requirements:
- basic_questions must vary openers, phrasing, and word choice across the three items; do not reuse the same starter or sentence pattern.
- Across different sections, avoid repeating identical question phrasings or starters consecutively.
- validation_question should begin with a different type of verification phrases (e.g., “Does this”,“Is it true”, "Can it be", “Is it accurate” and change accordingly), but vary the opener and wording for phrases ; do not reuse the same phrase everytime or rely only on these examples.
Do NOT use repeatedly same opener and phraser. Randomize opener selection and wording; avoid consecutive reuse of the same opener. Permit occasional repeats only with low probability (e.g., <10%) and never back-to-back.
- deepdive_question should be more probing and specific than basic_questions, using section_text keywords when available.

Guidelines:
- Ensure phrasing is natural and clinician-appropriate.
- {("Use the keywords meaningfully within the questions." if section_text else "Focus on the diagnosis_title to generate meaningful questions.")}
- Avoid starting questions with common phrases every time used in previous examples.
- Below are the Few-shot examples to illustrate the expected output format and for reference.

Input:
{{
  "diagnosis_title": "{diagnosis_title}",
  "section_name": "{section_name}",
  "section_text": {json.dumps(section_text)}
}}

Output format:
{{
  "diagnosis_title": "{diagnosis_title}",
  "section": "{section_name}",
  "items": [
    {{
      "basic_questions": [
        "<Question 1>",
        "<Question 2>",
        "<Question 3>"
      ],
      "validation_question": "<Single validation question>",
      "deepdive_question": "<Single deeper clinical question>"
    }}
  ]
}}

Few-shot example (FORMAT ONLY — do not copy wording):
Input:
{{
 "diagnosis_title":"Hypertension",
 "section_name":"background",
 "section_text":"Persistent elevated blood pressure; risk factors include obesity, high sodium diet, and sedentary lifestyle."
}}
Output:
{{
 "diagnosis_title":"Hypertension",
 "section":"background",
 "items":[{{
   "basic_questions":[
     "How long has elevated blood pressure been documented?",
     "Which risk factors — obesity and high sodium diet — are present?",
     "What aspects of sedentary lifestyle are recorded as contributing factors?"
   ],
   "validation_question":"Does evidence support an association between elevated blood pressure and obesity, high sodium diet, sedentary lifestyle?",
   "deepdive_question":"For Hypertension, how do obesity, high sodium diet, sedentary lifestyle shape thresholds for initiating therapy?"
 }}]
}}
"""

def combined_prompt(diagnosis_title: str, combined_label: str, combined_text: str) -> str:
    return f"""
Role: You are an expert in generating clinician-focused medical questions.
Task: Create ONE concise combined clinical question that jointly considers the two sections under the SAME diagnosis title.

Rules:
- Preserve diagnosis_title EXACTLY as provided.
- Do NOT include section names verbatim inside the question.
- If combined_text is available, use 2 to 3 salient keywords EXACTLY  to generate the combined_question.
- If combined_text is empty, use ONLY the diagnosis_title (preserve words/spellings).
- The question must be realistic, clinician-focused, and actionable.
- Do NOT include lists, quotes, code fences, or markdown.
- Return ONLY a valid JSON object with EXACT keys: diagnosis_title, combined_section, items.
- items must be an array with ONE object having EXACT key: combined_question.
- use different phrasing/openers/prefixes dont repeat same patterns from previous examples change after each question generation.
- combined_question must be a single combined clinical question that integrates concepts from combined sections.

Guidelines:
- Ensure phrasing is natural and clinician-appropriate.
- {("Use the keywords meaningfully within the questions." if combined_text else "Focus on the diagnosis_title to generate meaningful questions.")}
- Avoid starting questions with common phrases every time used in previous examples.
- Below are the Few-shot examples to illustrate the expected output format and for reference.

Input:
{{
  "diagnosis_title": "{diagnosis_title}",
  "combined_section": "{combined_label}",
  "combined_text": {json.dumps(combined_text)}
}}

Output format:
{{
  "diagnosis_title": "{diagnosis_title}",
  "combined_section": "{combined_label}",
  "items": [
    {{
      "combined_question": "<Single combined question>"
    }}
  ]
}}

Few-shot example (FORMAT ONLY — do not copy wording):
Input:
{{
  "diagnosis_title": "EXCESSIVE SLEEPINESS",
  "combined_section": "evaluation + management",
  "combined_text": "STOP-BANG; Epworth Sleepiness Scale; consider CPAP; rule out secondary causes"
}}
Output:
{{
  "diagnosis_title": "EXCESSIVE SLEEPINESS",
  "combined_section": "evaluation + management",
  "items": [
    {{
      "combined_question": "How should clinicians evaluate and manage excessive sleepiness considering STOP-BANG and Epworth scores?"
    }}
  ]
}}
"""