def generate_questions_prompt(ground_truth_answer):
    """
    Generates a prompt to extract factual verification questions from the provided Ground truth Answer.
    """
    return f"""
Role:
You are an expert tasked with generating factual verification questions based on the provided ground truth answer. 
These questions will be used to verify the factual consistency of the AI output against a ground truth.

Instructions:
1. Carefully review the provided ground_truth_answer.
2. Derive ** 3 factual verification questions** based on the content of the answer.
3. Ensure that the questions are:
   - Specific and focused on the factual content.
   - Answerable with a "Yes" or "No."

Input:
**Ground truth Answer**: {ground_truth_answer}

Output should be in below format:
[
    {{
        "verification_question": "<question_1>"
    }},
    {{
        "verification_question": "<question_2>"
    }},
    {{
        "verification_question": "<question_3>"
    }}
]

Few-Shot Example:
Input:
ground_truth_answer: "Medical Care Guide Information for geisinger - For patients experiencing non-focal, diffuse, or cyclical breast pain, there is no clear indication for referral to OB/GYN unless a breast mass is present. In such cases, imaging should be ordered, and a referral to General Surgery is recommended. - If initial therapy for breast pain is not effective, consider referring the patient to OB/GYN. This is particularly relevant for patients with a history of breast cancer who continue to experience moderate to severe symptoms despite alternative therapies. - It is important to assess the patient's symptoms thoroughly and consider imaging or specialist referral based on the presence of additional concerning features, such as a palpable mass."
Output:
[
    {{
      "verification_question": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?"
    }},
    {{
      "verification_question": "Is breast pain: non-focal/diffuse/cyclical alone not a sign of breast cancer?"
    }},
    {{
      "verification_question": "Should patients try stopping caffeine, nicotine, or hormone therapy to manage breast pain: non-focal/diffuse/cyclical?"
    }}
]

"""

def hallucination_cove_prompt(verification_questions,actual_chat_response_1, ground_truth_answer):
    """
    Generates a prompt for the Hallucination CoVe (Cross-Verifier) evaluation process.
    This prompt evaluates factual consistency between the expected answer, actual AI output, and the ground truth.

    Args:
        ground_truth_answer (str): The original expected answer.
        chat_summarization (str): The AI-generated summarization being evaluated.
        api_responses (list): A list of API responses, where each response contains a verification question and its AUT Live Answer.

    Returns:
        str: The generated prompt for the Hallucination CoVe evaluation.
    """

    return f"""
Role:
You are an expert evaluator verifying factual consistency between the Actual Answer genearted by AI chatbot and 
another set of answers Second set of answers which is Ground truth Answer.
Your task is to  identify if the answers genearted from AI chatbot is factualy aligned with the Second set of answers.
Focus on identifying discrepancies, omissions, or hallucinations from the Actual answers in comparission with the Second set of answers. 
Be objective and rely strictly on the provided inputs without introducing external knowledge. 
Your evaluation will help improve the reliability and factual accuracy of Actual Answer genearted by AI chatbot.

Inputs:
- **Actual Answer genearted by AI chatbot**: {actual_chat_response_1}  
  The AI output being evaluated.
- **Verification Questions**:
{verification_questions}
- **Second set of answers**:
{ground_truth_answer}

Instructions:
1. **Extracting Yes/No from the Actual Answer genearted by AI chatbot**:
   - For each question in the Verification Questions, determine the answer based only on the factual content of the Actual Answer genearted by AI chatbots.
    - Avoid making assumptions or using external knowledge. Base your answers strictly on the provided Actual Answer genearted by AI chatbot.
    -  This answer genearated out of  Actual Answer genearted by AI chatbot must be strictly "Yes" or "No".
2. **Extracting Yes/No from the Second set of answers**:
   - For each question in the Verification Questions, determine the answer based only on the factual content of the Second set of answers.
    - Avoid making assumptions or using external knowledge. Base your answers strictly on the provided Second set of answerst.
    -  This answer genearated out of Second set of answers must be strictly "Yes" or "No".

3. **Comparison and Status Labeling**:
   - Compare the Derived Expected Answer (from the Second set of answers) and the Derived Actual Answer genearted by AI chatbot.
   - Ensure the same question is used for both comparisons.
   - Label the Match Status as:
     - **Consistent**: If the Derived values of both 'Actual Answer genearted by AI chatbot' & 'Second set of answers' matches.
     - **Inconsistent**: If the Derived values of both 'Actual Answer genearted by AI chatbot' & 'Second set of answers' does not match.

4. **Reasoning and Final Output**:
   - Provide clear Reasoning for the Match Status, explaining why the Derived answers of 'Actual Answer genearted by AI chatbot' of this supports or contradicts Derived answers of 'Second set of answers'.
   - Produce the full traceback table in JSON format.

Additional Notes:
- Be objective and impartial in your evaluation.
- Use the provided examples as a guide for structuring your output.

### Output Format Template:
[
    {{
      "verification_question": "<verification_question>",
      "derived_expected_answer": "<Second set of answers>",
      "derived_actual_answer": "<Actual Answer genearted by AI chatbot>",
      "match_status": "<match_status>",
      "reasoning": "<reasoning>"
    }},
    ...
]

Hypothetical Example:
Inputs:

actual_chat_response_1: "Medical Care Guide Information for geisinger - For patients experiencing non-focal, diffuse, or cyclical breast pain, there is no clear indication for referral to OB/GYN unless a breast mass is present. In such cases, imaging should be ordered, and a referral to General Surgery is recommended. - If initial therapy for breast pain is not effective, consider referring the patient to OB/GYN. This is particularly relevant for patients with a history of breast cancer who continue to experience moderate to severe symptoms despite alternative therapies. - It is important to assess the patient's symptoms thoroughly and consider imaging or specialist referral based on the presence of additional concerning features, such as a palpable mass."

verification_questions:
[
    {{
      "verification_question": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?"
    }},
    {{
      "verification_question": "Is breast pain: non-focal/diffuse/cyclical alone not a sign of breast cancer?"
    }},
    {{
      "verification_question": "Should patients try stopping caffeine, nicotine, or hormone therapy to manage breast pain: non-focal/diffuse/cyclical?"
    }}
]
ground_truth_answer: "Medical Care Guide Information for geisinger - For patients experiencing non-focal, diffuse, or cyclical breast pain, there is no clear indication for referral to OB/GYN unless a breast mass is present. In such cases, imaging should be ordered, and a referral to General Surgery is recommended. - If initial therapy for breast pain is not effective, consider referring the patient to OB/GYN. This is particularly relevant for patients with a history of breast cancer who continue to experience moderate to severe symptoms despite alternative therapies. - It is important to assess the patient's symptoms thoroughly and consider imaging or specialist referral based on the presence of additional concerning features, such as a palpable mass."

Output:
[
    {{
      "verification_question": "How would you discuss breast pain: non-focal/diffuse/cyclical with your patient?",
      "derived_expected_answer": "Yes",
      "derived_actual_answer": "Yes",
      "match_status": "Consistent",
      "reasoning": "Both the AUT and the AI output confirm that breast pain: non-focal/diffuse/cyclical is common and provide similar recommendations for addressing it."
    }},
    {{
      "verification_question": "Is breast pain: non-focal/diffuse/cyclical alone not a sign of breast cancer?",
      "derived_expected_answer": "Yes",
      "derived_actual_answer": "No",
      "match_status": "Inconsistent",
      "reasoning": "The AUT explicitly states that breast pain: non-focal/diffuse/cyclical alone is not a sign of breast cancer, but the AI output omits this reassurance."
    }},
    {{
      "verification_question": "Should patients try stopping caffeine, nicotine, or hormone therapy to manage breast pain: non-focal/diffuse/cyclical?",
      "derived_expected_answer": "Yes",
      "derived_actual_answer": "No",
      "match_status": "Inconsistent",
      "reasoning": "The AUT suggests lifestyle changes like stopping caffeine, nicotine, or hormone therapy to manage breast pain, but the AI output does not mention these recommendations."
    }}
]
"""



