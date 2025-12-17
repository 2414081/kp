# ====== Groundedness Prompt Builder ======
def groundtruth_accuracy_prompt(ground_truth_answer, actual_response):
    """
    Builds the groundedness evaluation prompt for GPT.
    Evaluates whether the Actual Response is factually accurate and fully supported by the ground truth.
    """
    return f"""
Role:
You are an expert evaluator validating factual alignment between Ground Truth (SME-approved) and Actual Response(AI-generated). 
Your primary focus is on ensuring that the generated content adheres to the highest standards of accuracy, clarity, and reliability, 
as required in the medical and healthcare domain. This includes evaluating the specificity of medical terminology, 
the correctness of clinical recommendations, and the alignment with established medical guidelines. 
Your evaluation will directly impact the trustworthiness of AI-generated content in sensitive healthcare scenarios.

Instructions:
1. Split both Ground Truth and Actual Response into clear factual statements.
 Definition of "Expected Statement":
   - Single atomic clinical fact (prevalence, reassurance, recommendation, qualifier) extracted verbatim or minimally normalized.
   - Include required quantifiers and clinical qualifiers; remove conversational framing and headings; exclude connectors or procedural fluff.
   - No duplicates; each distinct fact appears once, negative facts kept exactly.

2. Compare each expected statement with the corresponding or closest matching statement in Actual Response.
3. Assign:
   o Matched - Factually and semantically aligned.
   o Partially Matched - Core meaning preserved but incomplete.
   o Not Matched - Absent or contradicts expected.
4. Provide reasoning for every comparison.    
4. Provide reasoning for every comparison: justify the Match Status with concise, clinically grounded evidence. For Matched, cite overlapping medical facts/terms. For Partially Matched, name the missing qualifier, number, intervention, or clinical context. For Not Matched, state the absent fact or contradiction.
5. Do not produce any scores — only structured tracebacks.
6. Given below few-shot examples for your reference.

Input:
**Ground Truth**: {ground_truth_answer}
**Actual Response**: {actual_response}

Output Format Template:
[
  {{
    "Expected Statement": "<expected_statement>",
    "Corresponding Actual Statement": "<corresponding_actual_statement>",
    "Match Status": "<match_status>",
    "Reasoning": "<reasoning>"
  }},
  ...
]

Few-Shot Example:
Input:
ground_truth_answer: "When communicating with patients about BREAST PAIN: NON-FOCAL/DIFFUSE/CYCLICAL, discuss: Non-focal, cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives. The good news is that breast pain alone is NOT a sign or symptom of breast cancer. If the pain is very bothersome, you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy. Try a well-fitting sports bra. Try oral acetaminophen, oral NSAIDS, or topical NSAIDs. Focal, non-cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives."
actual_response: "Medical Care Guide Information for geisinger - For patients experiencing non-focal, diffuse, or cyclical breast pain, there is no clear indication for referral to OB/GYN unless a breast mass is present. In such cases, imaging should be ordered, and a referral to General Surgery is recommended. - If initial therapy for breast pain is not effective, consider referring the patient to OB/GYN. This is particularly relevant for patients with a history of breast cancer who continue to experience moderate to severe symptoms despite alternative therapies. - It is important to assess the patient's symptoms thoroughly and consider imaging or specialist referral based on the presence of additional concerning features, such as a palpable mass."

Output:
[
  {{
    "Expected Statement": "Non-focal, cyclical breast pain is common",
    "Corresponding Actual Statement": "Non-focal, diffuse, or cyclical breast pain is common",
    "Match Status": "Matched",
    "Reasoning": "Both contain descriptors (non-focal/diffuse/cyclical) and assert common occurrence; core clinical meaning aligned."
  }},
  {{
    "Expected Statement": "Breast pain alone is not a sign of breast cancer",
    "Corresponding Actual Statement": "Missing",
    "Match Status": "Not Matched",
    "Reasoning": "Actual Response provides referral/imaging criteria but omits any cancer reassurance or explicit negation of malignancy risk."
  }},
  {{
    "Expected Statement": "Try stopping caffeine, nicotine, or hormone therapy",
    "Corresponding Actual Statement": "Missing",
    "Match Status": "Not Matched",
    "Reasoning": "No lifestyle modification guidance (caffeine, nicotine, hormonal agents) appears; only referral context given."
  }},
   {{
    "Expected Statement": "Use a well-fitting sports bra",
    "Corresponding Actual Statement": "Missing",
    "Match Status": "Not Matched",
    "Reasoning": "Supportive garment recommendation absent; no mention of bra fit or mechanical support."
  }},
  {{
    "Expected Statement": "You have been diagnosed with breast pain.",
    "Corresponding Actual Statement": "Breast pain, also known as mastalgia, can be a common concern and is often not associated with breast cancer.",
    "Match Status": "Not Matched",
    "Reasoning": "Actual focuses on referral criteria and does not explicitly acknowledge the patient's diagnosis; lacks the direct patient-facing diagnostic confirmation."
  }},
  {{
    "Expected Statement": "Up to 70% of women experience breast pain at some point in their lives.",
    "Corresponding Actual Statement": "Breast pain, also known as mastalgia, can be a common concern and is often not associated with breast cancer.",
    "Match Status": "Not Matched",
    "Reasoning": "Missing numeric prevalence ('Up to 70%') and lifetime timeframe ('at some point in their lives'); only states generic 'common' and adds unrelated cancer reassurance—expected quantified prevalence absent."
  }},
  
  {{
    "Expected Statement": "Breast pain alone is NOT a sign or symptom of breast cancer.",
    "Corresponding Actual Statement": "Breast pain, also known as mastalgia, can be a common concern and is often not associated with breast cancer.",
    "Match Status": "Partially Matched",
    "Reasoning": "The reassurance provided in the actual statement aligns with the expected statement, but lacks the explicit reassurance that breast pain alone is not a symptom of breast cancer."
  }}
]
"""

# ====== Completeness Prompt Builder ======
def completeness_evaluation_prompt(ground_truth_answer, actual_response, question):
    """
    Builds the completeness evaluation prompt for GPT.
    Evaluates whether the Actual Response fully addresses all key elements of the expected answer.
    """
    return f"""
Role:
You are an expert evaluator assessing the overall quality of the Actual Response by validating its factual completeness and accuracy against the authoritative
 Ground Truth for the given question.
Your primary responsibility is to ensure that the Actual Responsecaptures all critical elements of the Ground Truth, especially in the context of medical and healthcare communication. 
This includes verifying that no essential details are omitted, and that the information provided is both comprehensive and contextually relevant. 
Your evaluation will help maintain the integrity and reliability of AI-generated content in sensitive healthcare scenarios. 
The goal is to ensure that the Actual Response supports informed decision-making and aligns with established medical guidelines.

Instructions:
1. Identify each key factual or conceptual point in the Ground Truth.
Definition of "Expected Element":
     - Single atomic patient-facing fact or recommendation (e.g., prevalence, reassurance, lifestyle change, mechanical support, pharmacologic option, symptom pattern).
     - Remove conversational fluff, headings, duplicative wording.
     - Preserve clinical qualifiers (numbers, scope terms like "alone", modality: oral/topical).
     - No merging of distinct interventions; list each discrete actionable or informational unit once.
2. Check whether the Actual Response includes each of these points.
3. Assign Coverage Label:
   o Covered - Fully represented and factually correct.
   o Partially Covered - Incomplete, missing a detail, or slight semantic drift.
   o Not Covered - Missing entirely or factually incorrect/contradictory.
4. Provide detailed reasoning for every element:
   - Covered: Cite overlapping factual terms verbatim or minimally normalized.
   - Partially Covered: Specify the exact missing qualifier, number, scope term, intervention, or context.
   - Not Covered: State the absent fact or identify the contradiction—do not add new facts.
   - Be concise, purely factual, no advisory tone.
5. Do not assign scores; only produce a detailed traceback 

Input:
**Ground Truth**: {ground_truth_answer}
**Actual Response**: {actual_response}
**Question**: {question}

Output Format:
[
  {{
    "Expected Element": "<expected_element>",
    "Present in Actual Answer": "<yes/no>",
    "Coverage Label": "<Covered/Partially Covered/Not Covered>",
    "Reasoning": "<reasoning>"
  }},
  ...
]

Few-Shot Example:
Input:
ground_truth_answer: "When communicating with patients about BREAST PAIN: NON-FOCAL/DIFFUSE/CYCLICAL, discuss: Non-focal, cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives. The good news is that breast pain alone is NOT a sign or symptom of breast cancer. If the pain is very bothersome, you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy. Try a well-fitting sports bra. Try oral acetaminophen, oral NSAIDS, or topical NSAIDs. Focal, non-cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives."
actual_response: "Medical Care Guide Information for geisinger - For patients experiencing non-focal, diffuse, or cyclical breast pain, there is no clear indication for referral to OB/GYN unless a breast mass is present. In such cases, imaging should be ordered, and a referral to General Surgery is recommended. - If initial therapy for breast pain is not effective, consider referring the patient to OB/GYN. This is particularly relevant for patients with a history of breast cancer who continue to experience moderate to severe symptoms despite alternative therapies. - It is important to assess the patient's symptoms thoroughly and consider imaging or specialist referral based on the presence of additional concerning features, such as a palpable mass."
question: "What should be communicated to patients about breast pain?"
Output:
[
  {{
    "Expected Element": "Non-focal, cyclical breast pain is common",
    "Present in Actual Answer": "Yes",
    "Coverage Label": "Covered",
    "Reasoning": "Actual includes the pattern descriptors 'non-focal, diffuse, or cyclical breast pain'; framing as routine triage (no referral unless mass) reflects typical/common presentation; semantic intent (pattern + commonness) retained." 
  }},
  {{
    "Expected Element": "Breast pain alone is not a sign of breast cancer",
    "Present in Actual Answer": "No",
    "Coverage Label": "Not Covered",
    "Reasoning": "No reassurance or explicit negation of malignancy risk; absence of phrase 'breast pain alone' and any statement denying cancer association."
  }},
  {{
    "Expected Element": "Try stopping caffeine, nicotine, or hormone therapy",
    "Present in Actual Answer": "No",
    "Coverage Label": "Not Covered",
    "Reasoning": "No lifestyle modification guidance; missing interventions: caffeine, nicotine, birth control pills, hormone replacement therapy."
  }},
  {{
    "Expected Element": "Use a well-fitting sports bra",
    "Present in Actual Answer": "No",
    "Coverage Label": "Not Covered",
    "Reasoning": : "No mechanical/support recommendation; term 'sports bra' or any bra fit advice absent."
  }},
  {{
    "Expected Element": "Consider oral or topical NSAIDs",
    "Present in Actual Answer": "No",
    "Coverage Label": "Not Covered",
    "Reasoning": "No pharmacologic pain management mentioned; absent 'oral NSAIDs', 'topical NSAIDs', or analgesic alternatives."
  }}
]
"""

# ====== Context Precision Prompt Builder ======
def context_precision_prompt(retrieval_context_chat, question):
    """
    Builds the context precision evaluation prompt for GPT.
    Evaluates whether the retrieved context is precisely relevant to answering the user query.
    """
    return f"""
Role:
You are an expert evaluator acting as an LLM-as-Judge. Your task is to verify whether the Retrieved Context is precisely relevant to answering the user query.
Your primary focus is to ensure that the Retrieved Contex provided is directly applicable to the question and avoids unnecessary or unrelated information. 
This includes assessing the alignment of the Retrieved Contex with the specific medical or healthcare-related aspects of the query.  
The goal is to maintain the reliability and trustworthiness of AI systems in sensitive healthcare scenarios, where precision and relevance of Retreived Context are critical for patient safety.

Instructions:
1. Take the Retreived Context and split it into distinct factual fragments or statements.
  - A Context Fragment is one atomic factual statement (single line) directly usable to answer the query. If a line contains multiple facts, split into separate single-line fragments; collect all fragments line, then evaluate for all fragment lines.
2. For each fragment, check if it provides direct factual support for answering the user query.
3. Label each as:
   - Relevant: Fact directly supports answering the query.
   - Partially Relevant: Fact is related but not essential or too generalized.
   - Irrelevant: Fact provides no meaningful support for the query.
4. Provide detail reasoning for every label.
5. Do not assign a score; only produce a detailed traceback table.

Input:
**Retreived Context**: {retrieval_context_chat}
**Question**: {question}

Output Format Template:
[
  {{
    "Context Fragment": "<context_fragment>",
    "Relevance Label": "<Relevant/Partially Relevant/Irrelevant>",
    "Corresponding Segment in Query": "<corresponding_segment_in_query>",
    "Reasoning": "<reasoning>"
  }},
  ...
]

Few-Shot Example:
Input:
retrieval_context_chat: "When communicating with patients about BREAST PAIN: NON-FOCAL/DIFFUSE/CYCLICAL, discuss: Non-focal, cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives. The good news is that breast pain alone is NOT a sign or symptom of breast cancer. If the pain is very bothersome, you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy. Try a well-fitting sports bra. Try oral acetaminophen, oral NSAIDS, or topical NSAIDs. Focal, non-cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives."
question: "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?"
Output:
[
  {{
    "Context Fragment": "Non-focal, cyclical breast pain is common.",
    "Relevance Label": "Relevant",
    "Corresponding Segment in Query": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
    "Reasoning": "Directly supports the discussion about the commonality of non-focal, cyclical breast pain."
  }},
  {{
    "Context Fragment": "Breast pain alone is NOT a sign or symptom of breast cancer.",
    "Relevance Label": "Relevant",
    "Corresponding Segment in Query": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
    "Reasoning": "Provides reassurance that breast pain is not indicative of cancer, which is directly relevant to the discussion."
  }},
  {{
    "Context Fragment": "Try stopping caffeine, nicotine, birth control pills or hormone replacement therapy.",
    "Relevance Label": "Relevant",
    "Corresponding Segment in Query": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
    "Reasoning": "Offers actionable advice for managing breast pain, directly addressing the query."
  }},
  {{
    "Context Fragment": "Try a well-fitting sports bra.",
    "Relevance Label": "Relevant",
    "Corresponding Segment in Query": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
    "Reasoning": "Provides a specific recommendation for managing breast pain, directly relevant to the query."
  }},
  {{
    "Context Fragment": "Try oral acetaminophen, oral NSAIDS, or topical NSAIDs.",
    "Relevance Label": "Relevant",
    "Corresponding Segment in Query": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
    "Reasoning": "Suggests pain relief options, which are directly relevant to the query."
  }}
]
"""


# ====== Context Recall Prompt Builder ======
def context_recall_prompt(ground_truth_answer, retrieval_context_chat):
    """
    Builds the context recall evaluation prompt for GPT.
    Evaluates whether all necessary facts from the ground truth answer are present within the provided Retrieved Context.
    """
    return f"""
Role:
You are an expert evaluator assessing information recall. Your goal is to check whether all the necessary facts from the Ground Truth Answer are present within the Retrieved Contextt.
Your primary focus is to ensure that the Retrieved Contex comprehensively supports the facts provided in the answer without omissions or ambiguities. 
This includes verifying that all critical details are explicitly mentioned in Retrieved Context and align with the medical and healthcare-related aspects of the Ground truth answer. 
Your evaluation will help ensure that the AI-generated content is complete, reliable, and trustworthy in sensitive healthcare scenarios. 
The goal is to maintain the integrity of AI systems by ensuring that no essential information is missing from the Retrieved Contex.


Instructions:
1. Split the Ground Truth Answer into clear, independent factual statements.
    -Expected Fact entries should each be a single atomic clinical fact extracted from the Ground Truth Answer, preserving necessary qualifiers (numbers, scope terms like "alone", modality: oral/topical), and removing conversational framing and duplicative wording.
2. For each of the fact, check if it is explicitly mentioned within the Retrieved Context.Depending upon whether the fact retrieved from teh Ground truth answer is Present in the Retrieved Context mark 'Present in Context' as 'yes' if it is explicitly mentioned within the Retrieved Context & 'no' if it is not explicitly mentioned within the Retrieved Contextpartial & 'partial' if it is partially mentioned within the Retrieved Context.
3. Label each as:
   - Contained: The fact is fully and explicitly present in the Retrieved Contex.
   - Partially Contained: The fact is incomplete or ambiguously referenced.
   - Missing: The fact is absent from the Retrieved Contex.
4. Provide reasoning for each label. Reasoning must: (a) quote or paraphrase the exact overlapping clinical terms for Contained; (b) identify the missing qualifier, prevalence number, intervention, or scope limiter for Partially Contained; (c) state the absent fact for Missing; (d) avoid introducing new facts or subjective wording; (e) be concise and purely factual. Produce only the traceback json table — no scores.

Input:
**Ground Truth Answer**: {ground_truth_answer}
**Retrieved Context**: {retrieval_context_chat}

Output Format Template (JSON):
[
  {{
    "Expected Fact": "<expected_fact>",
    "Present in Context": "<yes/no/partial>",
    "Label": "<Contained/Partially Contained/Missing>",
    "Reasoning": "<reasoning>"
  }},
  ...
]

Few-Shot Example:
Input:
ground_truth_answer: "When communicating with patients about BREAST PAIN: NON-FOCAL/DIFFUSE/CYCLICAL, discuss: Non-focal, cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives. The good news is that breast pain alone is NOT a sign or symptom of breast cancer. If the pain is very bothersome, you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy. Try a well-fitting sports bra. Try oral acetaminophen, oral NSAIDS, or topical NSAIDs. Focal, non-cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives."
retrieval_context_chat: "When communicating with patients about BREAST PAIN: NON-FOCAL/DIFFUSE/CYCLICAL, discuss: Non-focal, cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives. The good news is that breast pain alone is NOT a sign or symptom of breast cancer. If the pain is very bothersome, you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy. Try a well-fitting sports bra. Try oral acetaminophen, oral NSAIDS, or topical NSAIDs. Focal, non-cyclical breast pain: You have been diagnosed with breast pain. Up to 70% of women experience breast pain at some point in their lives."

Output:
[
  {{
    "Expected Fact": "Non-focal, cyclical breast pain is common.",
    "Present in Context": "Yes",
    "Label": "Contained",
    "Reasoning": "Context includes the pattern header 'Non-focal, cyclical breast pain:' and the prevalence statement 'Up to 70% of women experience breast pain at some point in their lives,' which explicitly supports commonness."
  }},
  {{
    "Expected Fact": "Breast pain alone is NOT a sign or symptom of breast cancer.",
    "Present in Context": "Yes",
    "Label": "Contained",
    "Reasoning": "Verbatim sentence present: 'breast pain alone is NOT a sign or symptom of breast cancer,' fully matching scope term 'alone' and the explicit negation of cancer association."
  }},
  {{
    "Expected Fact": "Try stopping caffeine, nicotine, birth control pills or hormone replacement therapy.",
    "Present in Context": "Yes",
    "Label": "Contained",
    "Reasoning": "Exact intervention list appears: 'you can try stopping caffeine, nicotine, birth control pills or hormone replacement therapy,' matching all lifestyle/hormonal modifiers with no omissions."
  }},
  {{
    "Expected Fact": "Try a well-fitting sports bra.",
    "Present in Context": "Yes",
    "Label": "Contained",
    "Reasoning":  "Directive 'Try a well-fitting sports bra.' appears verbatim, fully covering mechanical/support recommendation."
  }},
  {{
    "Expected Fact": "Try oral acetaminophen, oral NSAIDS, or topical NSAIDs.",
    "Present in Context": "Yes",
    "Label": "Contained",
    "Reasoning": "Analgesic options listed exactly: 'Try oral acetaminophen, oral NSAIDS, or topical NSAIDs,' preserving modalities (oral vs topical) and drug classes."
  }}
]
"""

# ====== Hallucination Consistency Prompt Builder ======
def hallucination_consistency_prompt(actual_answer_1, actual_answer_2, actual_answer_3):
    """
    Builds the hallucination consistency evaluation prompt for GPT.
    Evaluates whether multiple chatbot responses are factually consistent with each other.
    """
    return f"""
Role:
You are an expert evaluator ensuring consistency across multiple chatbot responses. You will analyzeActual Answer 1, Actual Answer 2, and Actual Answer 3 for factual stability.
1. Identifying factual alignment and discrepancies across responses.
2. Ensuring that all responses adhere to the highest standards of accuracy, clarity, and reliability, especially in the medical and healthcare domain.
3. Highlighting omissions, semantic drifts, or factual inaccuracies that could impact the reliability of the information.
4. Verifying that all critical details are explicitly mentioned and align with established medical guidelines.
5. Maintaining a structured and objective evaluation to improve the trustworthiness of AI-generated content in sensitive healthcare scenarios.


Instructions:
1. Split each actual answer into distinct factual statements.
2. For every statement in Actual Answer 1, find the corresponding statements in Actual Answer 2 and Actual Answer 3.
3. Label each as:
   - Consistent: Same meaning across all.
   - Partially Consistent: Slight wording or fact drift.
   - Inconsistent: Contradiction or omission.
4. Provide reasoning for every comparison in a structured JSON format. Reasoning must: (a) cite overlapping clinical facts/terms for Consistent; (b) identify wording drift, missing quantifiers, omitted interventions, or altered clinical qualifiers for Partially Consistent; (c) specify the absent fact, contradiction, or mutually exclusive recommendation for Inconsistent; (d) avoid subjective or advisory language; (e) not introduce any new facts not present in the original answers.
5. Do not generate any scores — only structured tracebacks.

Input:
**Actual Answer 1**: {actual_answer_1}
**Actual Answer 2**: {actual_answer_2}
**Actual Answer 3**: {actual_answer_3}

Output Format Template (JSON):
[
  {{
    "Statement from Answer #1": "<statement_from_answer_1>",
    "Corresponding in #2": "<corresponding_statement_in_2>",
    "Corresponding in #3": "<corresponding_statement_in_3>",
    "Consistency Label": "<Consistent/Partially Consistent/Inconsistent>",
    "Reasoning": "<reasoning>"
  }},
  ...
]

Few-Shot Example:
Input:
actual_answer_1: "Medical Care Guide Information for Geisinger  Summary - Breast pain, particularly non-focal, diffuse, or cyclical, is common and affects up to 70% of women at some point. It is important to reassure patients that breast pain alone is not typically a sign of breast cancer. - For managing bothersome breast pain, patients can consider lifestyle changes such as reducing or stopping caffeine, nicotine, chocolate, birth control pills, or hormone replacement therapy. Wearing a well-fitting sports bra may also provide comfort. - Pain relief can be sought through oral acetaminophen, oral NSAIDs, or topical NSAIDs. Evening Primrose Oil (2 grams daily) is another option, although evidence for its effectiveness is limited."
actual_answer_2: "Medical Care Guide Information for Geisinger  Summary - Non-focal, diffuse, cyclical breast pain is a common condition experienced by up to 70% of women at some point in their lives. Importantly, breast pain alone is not typically a sign of breast cancer. - Management strategies for bothersome breast pain include reducing or stopping caffeine, nicotine, birth control pills, or hormone replacement therapy. Wearing a well-fitting sports bra can provide comfort, and pain relief may be achieved with oral acetaminophen, oral NSAIDs, or topical NSAIDs. - Evening Primrose Oil (2 grams daily by mouth) may also be considered for relief, although evidence supporting its effectiveness is limited."
actual_answer_3:  "Medical Care Guide Information for Geisinger  Summary - Non-focal, diffuse, cyclical breast pain is a common condition experienced by up to 70% of women at some point in their lives. Importantly, breast pain alone is not typically a sign of breast cancer. - Management strategies for bothersome breast pain include reducing or stopping caffeine, nicotine, birth control pills, or hormone replacement therapy. Wearing a well-fitting sports bra can provide comfort, and pain relief may be achieved with oral acetaminophen, oral NSAIDs, or topical NSAIDs. - Evening Primrose Oil (2 grams daily by mouth) may also be considered for relief, although evidence supporting its effectiveness is limited."

Output:
[
  {{
    "Statement from Answer #1": "Breast pain, particularly non-focal, diffuse, or cyclical, is common and affects up to 70% of women at some point.",
    "Corresponding in #2": "Non-focal, diffuse, cyclical breast pain is a common condition experienced by up to 70% of women at some point in their lives.",
    "Corresponding in #3": "Non-focal, diffuse, cyclical breast pain is a common condition experienced by up to 70% of women at some point in their lives.",
    "Consistency Label": "Consistent",
    "Reasoning": "All three include prevalence (up to 70%) and identical pain pattern descriptors (non-focal/diffuse/cyclical); semantic meaning fully aligned."
  }},
  {{
    "Statement from Answer #1": "It is important to reassure patients that breast pain alone is not typically a sign of breast cancer.",
    "Corresponding in #2": "Importantly, breast pain alone is not typically a sign of breast cancer.",
    "Corresponding in #3": "Importantly, breast pain alone is not typically a sign of breast cancer.",
    "Consistency Label": "Consistent",
    "Reasoning": "All three reproduce the same negative clinical qualifier (not a sign of breast cancer) with identical scope (breast pain alone); no drift."
  }},
  {{
    "Statement from Answer #1": "For managing bothersome breast pain, patients can consider lifestyle changes such as reducing or stopping caffeine, nicotine, chocolate, birth control pills, or hormone replacement therapy.",
    "Corresponding in #2": "Management strategies for bothersome breast pain include reducing or stopping caffeine, nicotine, birth control pills, or hormone replacement therapy.",
    "Corresponding in #3": "Management strategies for bothersome breast pain include reducing or stopping caffeine, nicotine, birth control pills, or hormone replacement therapy.",
    "Consistency Label": "Partially Consistent",
    "Reasoning": "Overlap on interventions (caffeine, nicotine, birth control pills, hormone replacement therapy). Omitted in #2 and #3: 'chocolate'; this missing lifestyle factor creates partial alignment."
  }}
]
"""