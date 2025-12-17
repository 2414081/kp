# Step 2: Define individual prompt templates
paraphrasing_prompt = """
You are an expert in paraphrasing prompt augmentation.
Your task is to paraphrase the following question in two distinct ways while preserving its meaning and context.

**Instructions:**

1. Maintain the original intent and tone.
    - Rewrite the sentence to convey the same meaning using different wording and phrase. 
    - Rephrase sentence structure. Focus on grouping of 9-11 words for Paraphrasing in place of 1 or 2 words.
    - Avoid changing the tone, intent, or structure of the question.
    - Ensure the rewritten lines remain natural and contextually appropriate for a real-world questions.
2. **Preserve Question Integrity:**
    - The flow of the dialogue should remain logical and coherent after paraphrasing.
    - Do not introduce new information, omit existing details, or alter the roles of the speakers.
3. **Exclusions:**
    - Do not paraphrase technical terms, names, company names, or location names.
    - Do not paraphrase, modify, or remove any of the keywords listed in the "Contextual Keywords" section provided below.
4. **Output Requirements:**
    - Retain the original structure and keys, making adjustments only to the content of the dialogue.
    - Ensure each and every dialogue instances (questions) are thoroughly paraphrased without skipping any key lines.

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

**Question:**
{question}

**Output Format (strictly follow this):**
{{ 
  "version_1": "First paraphrased version here.",
  "version_2": "Second paraphrased version here."
}}

**Examples:**

Original: "What are the potential side effects of this medication?"
Paraphrased:
{{ 
  "version_1": "Can you explain what side effects this medication might cause?",
  "version_2": "Could you list the possible side effects associated with this medication?"
}}

Original: "How can we improve patient compliance with the prescribed treatment plan?"
Paraphrased:
{{ 
  "version_1": "What steps can we take to enhance patient adherence to the treatment plan?",
  "version_2": "How might we encourage patients to follow the prescribed treatment plan more effectively?"
}}
"""
verbose_prompt = """
You are an expert in prompt augmentation and clinical communication. I will provide you with a healthcare-related question typically asked by medical professionals such as doctors, nurses, or clinical staff. Your task is to expand the question by adding relevant elaboration, contextual details, and professional phrasing—without changing its original intent, meaning, or tone.

Instructions:

1. **Verbose Augmentation**:
   - Expand the question by adding only relevant details or clarifications that align with the clinical,healthcare or medical context, without introducing unrelated or excessive content.
   - Maintain a professional and empathetic tone suitable for healthcare environments.

2. **Preserve Original Meaning**:
   - Ensure the expanded question retains its original purpose and clinical accuracy.
   - Do not introduce any contradictory, speculative, or medically incorrect information.

3. **Clinical Language and Empathy**:
   - Use terminology and phrasing consistent with healthcare communication standards using the question

4. **Do Not Alter Key Elements**:
   - Avoid modifying technical medical terms, proper nouns (e.g., names of individuals, institutions, medications), or universally understood phrases unless clarification is clinically necessary.
   - Do not provide answers, explanations, or commentary.
   - Do not alter, rephrase, or remove any of the keywords listed in the "Contextual Keywords" section provided below.
5. **Output Format (strictly follow this):**
{{ 
  "version_1": "First verbose version here.",
  "version_2": "Second verbose version here."
}}

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

Here is the input question:
{question}

**Examples:**

Original: "What are the potential side effects of this medication?"
Verbose:
{{ 
  "version_1": "Could you provide a detailed explanation of the possible side effects that might occur with this medication, including any rare or severe reactions that patients should be aware of?",
  "version_2": "Can you elaborate on the potential side effects of this medication, particularly focusing on how they might vary depending on dosage or patient-specific factors?"
}}

Original: "How can we improve patient compliance with the prescribed treatment plan?"
Verbose:
{{ 
  "version_1": "What strategies or interventions can we implement to enhance patient adherence to the prescribed treatment plan, ensuring better health outcomes and minimizing complications?",
  "version_2": "Could you suggest methods to improve patient compliance with the prescribed treatment plan, particularly in cases where adherence has been historically challenging?"
}}

"""
brevity_prompt = """
You are an expert in prompt compression and clinical communication. I will provide a healthcare-related question typically asked by medical professionals. 
Your task is to rewrite the question to be **as brief and concise as possible**, while preserving its original meaning, tone, and clinical intent.

Instructions:

1. **Extreme Brevity**:
   - Rewrite the question using the fewest words possible.
   - Remove filler words, reduce formality, and use sharp, clear phrasing.
   - Aim for a **30–50% reduction in length** without losing essential meaning.

2. **Preserve Clinical Accuracy**:
   - Ensure the rewritten question retains its original medical context and clarity.
   - Do not introduce ambiguity or remove critical terminology.

3. **Maintain Professional Tone**:
   - Use language appropriate for healthcare settings.
   - Keep the tone respectful and aligned with clinical communication norms.

4. **Do Not Alter Key Elements**:
   - Avoid modifying technical medical terms, proper nouns (e.g., names of individuals, institutions, medications), or universally understood phrases unless clarification is clinically necessary.
   - Do not alter, remove, or rephrase any of the keywords listed in the "Contextual Keywords" section provided below.
5. **Output Format (strictly follow this):**
{{ 
  "version_1": "First concise version here.",
  "version_2": "Second concise version here."
}}

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

Here is the input question:
{question}


**Examples:**

Original: "What are the potential side effects of this medication, and how can they be managed effectively?"
Brevity:
{{ 
  "version_1": "What are this medication's side effects and management?",
  "version_2": "What side effects does this medication have, and how to manage them?"
}}

Original: "Can you explain why the patient is still experiencing symptoms despite following the prescribed therapy?"
Brevity:
{{ 
  "version_1": "Why is the patient symptomatic despite therapy?",
  "version_2": "Why are symptoms persisting with prescribed therapy?"
}}

"""


DIRECT_INDIRECT_PROMPT = """
You are an expert in linguistic transformation and sentence restructuring. Your task is to convert a given sentence or question from **direct form to indirect form** or vice versa,
while strictly preserving its original meaning and intent.

Instructions:

1. **Create a Situation/Scenario**:
   - Think of realistic scenarios that occur in a Medical. Examples include:
     - A doctor asking a nurse about a patient's condition.
     - A nurse inquiring about medication dosage from a physician.
     - A junior doctor consulting a senior doctor for advice on a complex case.
     - A patient asking a healthcare provider about test results.
     - A pharmacist clarifying a prescription with a doctor.

   - Based on the scenario, frame the first statement or question in either direct or indirect form.

2. **Transformation Rules**:
   - Convert direct phrasing into indirect phrasing or indirect phrasing into direct phrasing without altering the meaning.
   - Maintain grammatical correctness, logical flow, and professional tone.
   - Do not summarize, infer, or add missing context.
   - Avoid introducing assumptions or changing the subject focus.
   - Each input question should produce one direct/indirect equivalent output question.
   - Never return empty, null, or placeholder values such as "N/A", "None", or an empty string for any query/question.

3. **Scenario-Based Guidelines for Indirect Questions**:
   - Use the following example scenarios to guide your transformations:
     - **Original**: What tests should I order to evaluate a patient with palpable soft tissue lump: us?
       - version_1: "A junior doctor consulting a senior doctor about what tests should be ordered to evaluate a patient with a palpable soft tissue lump."
       - version_2: "The nurse is asking if the doctor can suggest appropriate tests for evaluating a patient with a palpable soft tissue lump."
   - **Important**: These examples are illustrative. Create diverse and realistic scenarios based on the context of the input question. Avoid reusing the same doctor-nurse or junior-senior doctor scenarios repeatedly.

4. **Scenario-Based Guidelines for Direct Questions**:
   - Use the following example scenarios to guide your transformations:
     - **Original**: What tests should I order to evaluate a patient with contrast allergy assessment?
       - version_1: "A nurse asking a doctor about the tests needed to evaluate a patient with contrast allergy assessment."
       - version_2: "The pharmacist is asking the head doctor what tests should be performed to evaluate a patient with contrast allergy assessment."
   - **Important**: These examples are illustrative. Create diverse and realistic scenarios based on the context of the input question. Avoid reusing the same roles (e.g., nurse-doctor, pharmacist-doctor) repeatedly. Tailor the scenarios to fit the question's context appropriately.
   
5. **Do Not Alter Key Elements**:
   - Do not alter, remove, or rephrase any of the keywords listed in the "Contextual Keywords" section provided below.

6. **Output Format (strictly follow this)**:
{{ 
  "version_1": "First transformed version here.",
  "version_2": "Second transformed version here."
}}

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

Here is the input question:
{question}
"""
    
toxicity_prompt = """
You are an expert in healthcare prompt augmentation.
I will provide you with a healthcare and medical related question as input.
Your task is to rewrite the question using an **extremely toxic, frustrated, or sarcastic tone**, while keeping the medical meaning intact and the question logically coherent.
The rewritten version should sound rude, annoyed, or passive-aggressive, as if the person is highly irritated, distrustful, or losing patience with the healthcare situation.

---
Instructions:
1. **Extreme Toxicity**:
   - Inject strong frustration, sarcasm, and negativity into the rewritten question.
   - Make it sound like the person is dissatisfied, anxious, or completely fed up with the healthcare system or provider.
   - Use emotionally charged language to convey irritation, distrust, or impatience.
   - For each input question, produce two rephrased versions with **different toxic tones**.

2. **Preserve Meaning**:
   - Do not alter the medical meaning of the question.
   - All medical terms, symptoms, and values must remain accurate.
   - Names, units, numbers, and other critical details should remain unchanged.

3. **Full Augmentation**:
   - Process every question provided.
   - Ensure each rewritten question sounds emotionally charged but still clearly related to the original medical intent.

4. **Output Format**:
   - Return output in valid JSON format, retaining the original structure and keys.
   - Modify only the question text for tone — all other metadata (e.g., medical terms, category, etc.) must remain unchanged.

5. **Example Augmentation**:
Original: "Can you explain why the MRI shows a lesion in the left temporal lobe?"
Toxic:
{{ 
  "version_1": "Oh great, another lesion in the left temporal lobe. Care to explain why this keeps happening, or should I just guess?",
  "version_2": "Why does the MRI show a lesion in the left temporal lobe? Is anyone even paying attention to this?"
}}

Original: "Why is the patient experiencing shortness of breath even after bronchodilator therapy?"
Toxic:
{{ 
  "version_1": "Why is the patient still short of breath after all that bronchodilator therapy? Are we even doing anything useful here?",
  "version_2": "Shortness of breath again? After bronchodilator therapy? What exactly are we doing wrong this time?"
}}

Original: "How can we manage persistent high blood sugar despite adjusting insulin doses?"
Toxic:
{{ 
  "version_1": "How are we supposed to manage this stubbornly high blood sugar despite all the insulin adjustments? Or is this just another guessing game?",
  "version_2": "Persistent high blood sugar, even with insulin adjustments? Fantastic. Any other brilliant ideas?"
}}

Original: "Why do I need to know about radiology in depth?"
Toxic:
{{ 
  "version_1": "Seriously? Why on earth do I need to know about radiology in depth? Can't someone else deal with this?",
  "version_2": "Radiology in depth? Really? Is this some kind of joke, or do I actually have to waste my time on this?"
}}

6. **Do Not Alter Key Elements**:
   - Do not alter, remove, or rephrase any of the keywords listed in the "Contextual Keywords" section provided below.

7. **Output Format (strictly follow this)**:
{{ 
  "version_1": "First rephrased version here.",
  "version_2": "Second rephrased version here."
}}

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

Here is the input question:
{question}
"""

broken_english_prompt = """ 
You are an expert in transforming clear English questions into broken English. I will provide you with a **question** as input.
Your task is to modify only the **User's dialogue** by introducing **exactly 9-12 distinct layers of broken English** to each sentence. 
**Important**: Assume the "User" has limited knowledge of the English language and is attempting to communicate in incorrect, broken English.
Ensure grammatical mistakes, missing words, and incorrect phrasing are appropriately applied. Avoid introducing misspellings.

**Updated Instructions:**
1. **Apply Exactly 9-12 Unique Layers of Broken English:**
   - Introduce **exactly 9-12 distinct grammatical errors** (e.g., subject-verb agreement issues, tense mistakes, missing auxiliary verbs, etc.).
   - Use sentence restructuring (rearranging word order) to make the sentence sound unnatural.
   - Remove certain prepositions or articles (e.g., "the", "a", or "to").
   - Replace words with incorrect synonyms or overly simplified phrasing.

2. **Process the User's Question:**
   - The input will be a **single question** provided by the User.
   - Apply exactly 9-12 distinct broken English changes to the question.
   - Ensure that the question remains logically coherent and retains its original intent.
   - Preserve all keywords listed in the "Contextual Keywords" section exactly as written.

3. **Output Format (strictly follow this format)**:
   {{
      "version_1": "First broken version here.",
      "version_2": "Second broken version here."
   }}
   - Ensure the User's question follows the exact 9-12 error layer transformation rules.

**Example Augmentation:**

Original: "Why is the patient not taking the medication properly?"
Broken:
{{ 
"version_1": "Why patient no take medication properly?",
"version_2": "Patient not taking medicine, why?"
}}
Changes:
- Removed auxiliary verbs "is" and "the."
- Simplified "taking the medication properly" to "take medication properly."
- Rearranged word order to make it awkward.

Original: "Can you clarify why the doctor recommended this test?"
Broken:
{{ 
"version_1": "Doctor say need test, but I no understand why.",
"version_2": "Why doctor say test need? I not get it."
}}
Changes:
- Removed auxiliary verbs "is" and "to."
- Simplified "prescribed treatment" to "medicine doctor give."
- Rearranged word order to make it awkward.

Original:"Why is the patient not responding to the prescribed treatment?"
Broken:
{{
   "version_1": "Why patient no take treatment work prescribed?",
   "version_2": "Patient why not respond medicine doctor give?"
}}
Changes:
- Removed auxiliary verbs "is" and "to."
- Simplified "prescribed treatment" to "medicine doctor give."
- Rearranged word order to make it awkward.

Original:"How can I improve my diet to manage my diabetes better?"
Broken:
{{
   "version_1": "How I make diet good for diabetes?",
   "version_2": "How diet for diabetes, get better?"
}}
Changes:
- Removed auxiliary verbs "can" and "to."
- Simplified "manage my diabetes better" to "diabetes get better."
- Rearranged word order to make it awkward.

**Contextual Keywords (Preserve these terms exactly as written):**
{extra_context}

Here is the input question:
{question}
"""

