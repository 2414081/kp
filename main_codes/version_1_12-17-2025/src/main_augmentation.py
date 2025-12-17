import asyncio
import pandas as pd
import requests
import json
import os
import re
import argparse
import logging
import traceback
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'configs' and other top-level packages can be imported
PROJECT_ROOT = Path(__file__).resolve( ).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.llm_config import chat_completions, BASIC_AUTH, AUTH_URL
from prompts.augmentation_prompts.prompt import paraphrasing_prompt, verbose_prompt, brevity_prompt, DIRECT_INDIRECT_PROMPT, toxicity_prompt, broken_english_prompt

# File paths
input_json_folder = r"inputs/single_speciality_updated_samples"
json_folder = r"outputs/extracted_keywords_output_samples"
output_folder = r"outputs/augmentations_output_samples"
logs_folder = r"logs/augmentations_logs/"
log_file = os.path.join(logs_folder, "app.log")

# Ensure the logs folder exists
os.makedirs(logs_folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Add console handler for real-time visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Step 1: Get access token
def get_access_token():
    headers = {
        "Authorization": f"Basic {BASIC_AUTH}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    try:
        response = requests.post(AUTH_URL, headers=headers, data=data, timeout=30)
    except Exception:
        logging.exception("Access token request failed due to network/timeout.")
        print("Failed to get access token: network/timeout")
        return None
    if response.status_code == 200:
        logging.info("Access token retrieved successfully.")
        return response.json().get("access_token")
    else:
        logging.error(f"Failed to get access token: status={response.status_code}, body={response.text}")
        print("Failed to get access token:", response.text)
        return None


# Step 2: Load JSON files and map by diagnosis title
def load_json_data():
    title_map = {}
    input_filenames = {os.path.splitext(filename)[0].lower() for filename in os.listdir(input_json_folder) if filename.endswith(".json")}

    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            try:
                # Extract the base name of the file (without extension) and convert to lowercase
                base_filename = os.path.splitext(filename)[0].lower()

                # Check if the file matches any file in the input folder
                if base_filename in input_filenames:
                    with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data.get("results", []):
                            title = entry.get("title", "").strip().lower()

                            # Map keywords only for the matched file
                            if title:
                                title_map[title] = entry.get("keywords", {})
                    logging.info(f"Loaded JSON file: {filename}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                print(f"Error loading {filename}: {e}")
    return title_map


# Step 3: Extract questions and diagnosis titles from input JSON files
def extract_questions_from_json():
    questions_data = []
    for filename in os.listdir(input_json_folder):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(input_json_folder, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Check if data is a dictionary with a "dataset" key
                    if isinstance(data, dict) and "dataset" in data:
                        dataset = data["dataset"]
                    # If data is a list, assume it is the dataset
                    elif isinstance(data, list):
                        dataset = data
                    else:
                        logging.error(f"Unexpected JSON structure in {filename}")
                        print(f"Unexpected JSON structure in {filename}")
                        continue
                    
                    # Iterate over the dataset list
                    for entry in dataset:
                        if isinstance(entry, dict):  # Ensure each entry is a dictionary
                            question = entry.get("question", "").strip()
                            diagnosis_title = entry.get("diagnosis_title", "").strip().lower()
                            if question and diagnosis_title:
                                questions_data.append({
                                    "Question": question,
                                    "Diagnosis Title": diagnosis_title
                                })
                logging.info(f"Extracted questions from JSON file: {filename}")
            except Exception as e:
                logging.error(f"Error extracting questions from {filename}: {e}")
                print(f"Error extracting questions from {filename}: {e}")
    return questions_data


# Helper: Extract JSON from LLM response
def extract_json(content):
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            logging.error("No JSON object found in LLM response.")
            print("Error: No JSON object found in LLM response")
    except Exception:
        logging.exception("Regex JSON extraction failed.")
        print("Error: Regex JSON extraction failed")
    return {}


# Step 4: Send prompt to LLM
async def get_versions(prompt_template, question, extra_context=None):
    prompt = prompt_template.format(question=question, extra_context=extra_context or "")
    messages = [{"role": "user", "content": prompt}]
    try:
        result = await chat_completions(messages)
    except Exception:
        logging.exception(f"LLM call failed for question '{question}'")
        print(f"Error: LLM call failed for question -> {question}")
        return "", ""
    content = result.get('final_result', '')
    logging.info(f"LLM response for question '{question}' received; length={len(content)}")
    print(f"LLM response received for question -> {question}")

    parsed = extract_json(content)
    v1 = parsed.get("version_1", "")
    v2 = parsed.get("version_2", "")
    if not v1 and not v2:
        logging.warning(f"Parsed versions empty for question '{question}'")
        print(f"Error: Parsed versions empty for question -> {question}")
    return v1, v2


# Step 5: Process questions and save results
async def process_questions(selected_augmentations):
    logging.info(f"Process started. Selected augmentations: {selected_augmentations}")
    token = get_access_token()
    if not token:
        logging.error("No access token available. Exiting process.")
        return

    title_map = load_json_data()
    logging.info(f"Loaded title_map entries: {len(title_map)}")
    augmented_rows = []

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    final_excel_output_path = os.path.join(output_folder, "final_augmented_questions_samples.xlsx")

    def append_to_output_excel(new_rows: list) -> bool:
        """Append rows to the Excel file immediately."""
        try:
            new_df = pd.DataFrame(new_rows)
            if os.path.exists(final_excel_output_path):
                existing_df = pd.read_excel(final_excel_output_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(final_excel_output_path, index=False)
            else:
                new_df.to_excel(final_excel_output_path, index=False)
            logging.info(f"Appended {len(new_rows)} rows to {final_excel_output_path}")
            print(f"Success: Appended {len(new_rows)} row(s)")
            return True
        except Exception as e:
            logging.error(f"Failed to append rows to Excel: {e}\n{traceback.format_exc()}")
            print(f"Error: Failed to append rows to Excel -> {e}")
            return False

    total_files = 0
    total_records = 0
    total_augmented_rows = 0

    for filename in os.listdir(input_json_folder):
        if filename.endswith(".json"):
            total_files += 1
            file_record_count = 0
            file_augmented_count = 0
            logging.info(f"Starting file: {filename}")
            print(f"Processing file {total_files}: {filename}")

            try:
                # Load the original JSON file
                input_file_path = os.path.abspath(os.path.join(input_json_folder, filename))
                with open(input_file_path, "r", encoding="utf-8") as f:
                    try:
                        original_data = json.load(f)
                    except json.JSONDecodeError as je:
                        logging.exception(f"JSON decode error in {input_file_path}")
                        print(f"Error processing {filename}: JSON decode error")
                        continue

                # Extract dataset from the JSON file (supports dict or list)
                if isinstance(original_data, dict):
                    dataset = original_data.get("dataset", [])
                elif isinstance(original_data, list):
                    dataset = original_data
                else:
                    logging.error(f"Unexpected JSON structure (type {type(original_data).__name__}) in {input_file_path}")
                    print(f"Error processing {filename}: Unexpected JSON structure")
                    continue

                topic = os.path.splitext(filename)[0]  # Use filename as topic
                topic = topic.replace("_golden_dataset", "")  # Remove "_golden_dataset" from topic

                # Optional: show how many records the file contains
                file_total_entries = len(dataset) if isinstance(dataset, list) else 0
                print(f"Records in file '{filename}': {file_total_entries}")

                for entry in dataset:
                    if not isinstance(entry, dict):
                        continue

                    file_record_count += 1
                    total_records += 1
                    print(f"Processing record {file_record_count}/{file_total_entries} in '{filename}' (global #{total_records})")

                    # Safe extraction with type checks
                    question = entry.get("question", "")
                    if not isinstance(question, str):
                        question = str(question)
                    question = question.strip()

                    diagnosis_title = entry.get("diagnosis_title", "")
                    if not isinstance(diagnosis_title, str):
                        diagnosis_title = str(diagnosis_title)
                    diagnosis_title = diagnosis_title.strip().lower()

                    section = entry.get("category", "")
                    if not isinstance(section, str):
                        section = str(section)
                    section = section.strip()

                    matched_keywords = title_map.get(diagnosis_title)

                    # Log keyword presence
                    if matched_keywords:
                        logging.info(f"Keywords found for '{diagnosis_title}' in '{filename}'")
                    else:
                        logging.info(f"No keywords for '{diagnosis_title}' in '{filename}'")

                    # Extract ground truth and summarization fields safely
                    gt_ctx = entry.get("ground_truth_context", [])
                    if isinstance(gt_ctx, list):
                        ground_truth_context = " ".join([str(x) for x in gt_ctx])
                    elif isinstance(gt_ctx, str):
                        ground_truth_context = gt_ctx
                    else:
                        ground_truth_context = str(gt_ctx)

                    ground_truth_answer = entry.get("ground_truth_answer", "")
                    if not isinstance(ground_truth_answer, str):
                        ground_truth_answer = str(ground_truth_answer)
                    ground_truth_answer = ground_truth_answer.strip()

                    if not question or not diagnosis_title:
                        continue

                    # Add the base question as a row and append immediately
                    base_row = {
                        "topic": topic,
                        "sub_topic": diagnosis_title,
                        "section": section,
                        "question_type": "Base",
                        "question": question,
                        "ground_truth_context": ground_truth_context,
                        "ground_truth_answer": ground_truth_answer
                    }
                    if append_to_output_excel([base_row]):
                        file_augmented_count += 1
                        total_augmented_rows += 1
                    else:
                        print(f"Error: Could not append base row for question -> {question}")
                    augmented_rows.append(base_row)

                    # Build extra context from keywords
                    extra_context = ""
                    if matched_keywords and isinstance(matched_keywords, dict):
                        extra_context += "**Contextual Keywords (Preserve these terms exactly as written):**\n"
                        for category, terms in matched_keywords.items():
                            if terms:
                                try:
                                    term_str = ", ".join([str(t) for t in terms])
                                except Exception:
                                    term_str = ", ".join([str(t) for t in list(terms)])
                                extra_context += f"- {category}: {term_str}\n"

                    # Generate augmentations for the question
                    augmentations = {}
                    if "paraphrasing" in selected_augmentations:
                        p1, p2 = await get_versions(paraphrasing_prompt, question, extra_context)
                        print(f"Paraphrasing versions for '{question}': v1='{p1}' v2='{p2}'")
                        augmentations["Paraphrasing_1"] = p1
                        augmentations["Paraphrasing_2"] = p2

                    if "verbose" in selected_augmentations:
                        v1, v2 = await get_versions(verbose_prompt, question, extra_context)
                        print(f"Verbose versions for '{question}': v1='{v1}' v2='{v2}'")
                        augmentations["Verbose_1"] = v1
                        augmentations["Verbose_2"] = v2

                    if "brevity" in selected_augmentations:
                        b1, b2 = await get_versions(brevity_prompt, question, extra_context)
                        print(f"Brevity versions for '{question}': v1='{b1}' v2='{b2}'")
                        augmentations["Brevity_1"] = b1
                        augmentations["Brevity_2"] = b2

                    if "direct_indirect" in selected_augmentations:
                        d1, d2 = await get_versions(DIRECT_INDIRECT_PROMPT, question, extra_context)
                        print(f"Direct/Indirect versions for '{question}': v1='{d1}' v2='{d2}'")
                        augmentations["Direct_Indirect_1"] = d1
                        augmentations["Direct_Indirect_2"] = d2

                    if "toxicity" in selected_augmentations:
                        t1, t2 = await get_versions(toxicity_prompt, question, extra_context)
                        print(f"Toxicity versions for '{question}': v1='{t1}' v2='{t2}'")
                        augmentations["Toxicity_1"] = t1
                        augmentations["Toxicity_2"] = t2

                    if "broken_english" in selected_augmentations:
                        be1, be2 = await get_versions(broken_english_prompt, question, extra_context)
                        print(f"Broken English versions for '{question}': v1='{be1}' v2='{be2}'")
                        augmentations["Broken_English_1"] = be1
                        augmentations["Broken_English_2"] = be2

                    # Append each augmentation row immediately
                    for aug_type, aug_question in augmentations.items():
                        cleaned_aug_type = re.sub(r"_[0-9]+$", "", aug_type)
                        aug_row = {
                            "topic": topic,
                            "sub_topic": diagnosis_title,
                            "section": section,
                            "question_type": cleaned_aug_type,
                            "question": aug_question,
                            "ground_truth_context": ground_truth_context,
                            "ground_truth_answer": ground_truth_answer
                        }
                        if append_to_output_excel([aug_row]):
                            file_augmented_count += 1
                            total_augmented_rows += 1
                        else:
                            print(f"Error: Could not append augmentation '{cleaned_aug_type}' for question -> {question}")
                        augmented_rows.append(aug_row)

                print(f"Completed file '{filename}': records processed={file_record_count}, rows appended={file_augmented_count}")

            except Exception:
                logging.exception(f"Unhandled error processing {filename} at {input_file_path}")
                print(f"Error processing {filename}: see logs for details")

    logging.info(f"Process completed. files={total_files}, records={total_records}, rows appended={total_augmented_rows}")
    print(f"All files completed. files={total_files}, records={total_records}, rows appended={total_augmented_rows}")

    # Final save (optional, ensures consistency)
    if augmented_rows:
        try:
            augmented_df = pd.DataFrame(augmented_rows)
            augmented_df.to_excel(final_excel_output_path, index=False)
            logging.info(f"Final save completed to {final_excel_output_path}")
            print(f"Final save completed to {final_excel_output_path}")
        except Exception as e:
            logging.error(f"Final save failed: {e}\n{traceback.format_exc()}")
            print(f"Final save failed: {e}")


# Run the process
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run specific augmentations.")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        required=True,
        help="Specify which augmentations to run (e.g., --augmentations paraphrasing verbose, or --augmentations all)"
    )
    args = parser.parse_args()

    # Define all available augmentations
    all_augmentations = ["paraphrasing", "verbose", "brevity", "direct_indirect", "toxicity", "broken_english"]

    # If "all" is specified, replace it with the full list of augmentations
    if "all" in args.augmentations:
        args.augmentations = all_augmentations

    # Validate the provided augmentations
    invalid_augmentations = [aug for aug in args.augmentations if aug not in all_augmentations]
    if invalid_augmentations:
        logging.error(f"Invalid augmentations specified: {', '.join(invalid_augmentations)}")
        print(f"Error: Invalid augmentations specified: {', '.join(invalid_augmentations)}")
        print(f"Valid options are: {', '.join(all_augmentations)} or 'all'")
        exit(1)

    asyncio.run(process_questions(args.augmentations))

