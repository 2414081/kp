import pandas as pd
import os
import asyncio
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'configs' and other top-level packages can be imported
PROJECT_ROOT = Path(__file__).resolve( ).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
import logging
from logging.handlers import RotatingFileHandler
from configs.llm_config import chat_completions
from prompts.metric_prompts.prompt_2 import generate_questions_prompt

# Create logs directory (and subdirectory) if they don't exist
os.makedirs("logs/prep_data_generation_logs", exist_ok=True)

# Configure logging: rotating file + console, guard against duplicates
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = RotatingFileHandler(
        filename=r"logs/prep_data_generation_logs/prep_data_generation_2_log.log",
        mode="a",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Run started: prep_data_generation_2")

# Helper: Call LLM and get response
async def get_llm_response(prompt):
    """
    Send a prompt to the LLM and return the response.
    Retries on transient HTTP 503s and handles None responses safely.
    """
    messages = [{"role": "user", "content": prompt}]
    max_retries = 5
    backoff_sec = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            result = await chat_completions(messages)

            if result is None:
                logger.warning("LLM returned None (attempt %d).", attempt)
                print(f"WARN: LLM returned None (attempt {attempt}).")
            else:
                # Support both dict responses and plain strings
                if isinstance(result, dict):
                    final = result.get("final_result") or result.get("content") or ""
                else:
                    final = str(result)

                if final:
                    return final

                logger.warning("LLM returned empty final_result/content (attempt %d).", attempt)
                print(f"WARN: LLM returned empty final_result/content (attempt {attempt}).")

        except Exception as e:
            logger.warning(f"Error while calling LLM (attempt {attempt}): {e}", exc_info=True)
            print(f"WARN: Error while calling LLM (attempt {attempt}): {e}")

        # Exponential backoff before next retry
        await asyncio.sleep(backoff_sec)
        backoff_sec = min(backoff_sec * 2, 16)

    # All retries exhausted
    logger.error("LLM retries exhausted. Returning None.")
    print("ERROR: LLM retries exhausted. Returning None.")
    return None

# Helper: Clean LLM output
def _clean_llm_output(text: str) -> str:
    """
    Remove Markdown/quoted fences like ```json ... ``` or '''json ... '''
    and trim whitespace.
    """
    if text is None:
        return None
    s = str(text).strip()

    # Normalize line endings
    s = s.replace("\r\n", "\n")

    # Remove leading fenced language markers
    for fence in ("```json", "```", "'''json", "'''"):
        if s.lower().startswith(fence):
            s = s[len(fence):].lstrip()

    # Remove trailing fences
    for fence in ("```", "'''"):
        if s.endswith(fence):
            s = s[: -len(fence)].rstrip()

    # If the content is still fenced by first/last line, strip those lines
    lines = s.split("\n")
    if lines and lines[0].strip().lower() in ("```json", "```", "'''json", "'''"):
        lines = lines[1:]
    if lines and lines[-1].strip() in ("```", "'''"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

# Process the input column and generate prompts (incremental save)
async def process_excel(file_path, output_folder):
    """
    Incrementally generate chat_generated_questions.
    After each row is processed, persist the full dataframe to the output Excel.
    Resume on rerun by skipping rows already filled.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Input Excel not found: {file_path}")
            print(f"ERROR: Input Excel not found: {file_path}")
            return

        logger.info(f"Starting processing. Input: {file_path}, Output folder: {output_folder}")
        print(f"INFO: Starting processing. Input: {file_path}, Output: {output_folder}")
        df = pd.read_excel(file_path)
        logger.info(f"Total records loaded: {len(df)}")
        print(f"INFO: Total records loaded: {len(df)}")

        if "ground_truth_answer" not in df.columns:
            logger.error("'ground_truth_answer' column not found in the Excel file.")
            print("ERROR: 'ground_truth_answer' column not found in the Excel file.")
            return

        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "data_prep_2.xlsx")
        logger.info(f"Output will be saved to: {output_file}")
        print(f"INFO: Output will be saved to: {output_file}")

        # If previous output exists, merge existing processed columns
        if os.path.exists(output_file):
            try:
                existing = pd.read_excel(output_file)
                if len(existing) == len(df):
                    for col in ["chat_generated_questions"]:
                        if col in existing.columns:
                            df[col] = existing[col]
                    logger.info(f"Resumed previous progress from: {output_file}")
                    print(f"INFO: Resumed previous progress from: {output_file}")
                else:
                    msg = "Existing output length differs from input; ignoring previous partial results."
                    logger.warning(msg)
                    print(f"WARN: {msg}")
            except Exception as e:
                logger.warning(f"Could not load existing output for resume: {e}", exc_info=True)
                print(f"WARN: Could not load existing output for resume: {e}")

        # Ensure target column exists
        if "chat_generated_questions" not in df.columns:
            df["chat_generated_questions"] = None
            logger.info("Initialized 'chat_generated_questions' column.")
            print("INFO: Initialized 'chat_generated_questions' column.")

        gt_series = df["ground_truth_answer"]

        # Iterate rows; skip already processed
        for display_idx, (row_idx, value) in enumerate(gt_series.items(), start=1):
            try:
                if pd.isna(value) or not str(value).strip():
                    if pd.isna(df.at[row_idx, "chat_generated_questions"]):
                        df.at[row_idx, "chat_generated_questions"] = None
                    logger.info(f"Row {display_idx}: Empty ground_truth_answer. Skipped.")
                    print(f"INFO: Row {display_idx}: Empty ground_truth_answer. Skipped.")
                    continue

                existing_q = df.at[row_idx, "chat_generated_questions"]
                if existing_q:
                    logger.info(f"Row {display_idx}: Already processed. Skipping.")
                    print(f"INFO: Row {display_idx}: Already processed. Skipping.")
                    continue

                chat_prompt = generate_questions_prompt(value)
                logger.info(f"Row {display_idx}: Sending prompt to LLM.")
                print(f"INFO: Row {display_idx}: Sending prompt to LLM.")
                chat_resp = await get_llm_response(chat_prompt)

                if chat_resp is None:
                    logger.error(f"Row {display_idx}: LLM call failed. No response.")
                    print(f"ERROR: Row {display_idx}: LLM call failed. No response.")
                    continue

                cleaned = _clean_llm_output(chat_resp)
                logger.info(f"Row {display_idx}: LLM response received (length {len(cleaned)}).")
                print(f"INFO: Row {display_idx}: LLM response received. len={len(cleaned)}")

                # Option A: full response
                print("INFO: Row {display_idx}: LLM full response:")
                print(cleaned)

                # Option B: longer preview (e.g., 2000 chars)
                # print(f"INFO: Row {display_idx}: LLM response preview: {cleaned[:2000]}")

                # Option C: pretty-print JSON if it’s JSON
                # try:
                #     import json
                #     parsed = json.loads(cleaned)
                #     print("INFO: Row {display_idx}: LLM pretty JSON response:")
                #     print(json.dumps(parsed, indent=2))
                # except Exception:
                #     print(f"INFO: Row {display_idx}: LLM response preview: {cleaned[:1000]}")

                df.at[row_idx, "chat_generated_questions"] = cleaned

                # Incremental save after each processed row
                try:
                    df.to_excel(output_file, index=False)
                    logger.info(f"Row {display_idx}: Progress saved -> {output_file}")
                    print(f"INFO: Row {display_idx}: Progress saved -> {output_file}")
                    # Running counts
                    generated_so_far = sum(bool(x) for x in df["chat_generated_questions"])
                    print(f"INFO: Generated so far: {generated_so_far}/{len(df)}")
                except Exception as e:
                    logger.error(f"Row {display_idx}: Failed to save progress: {e}", exc_info=True)
                    print(f"ERROR: Row {display_idx}: Failed to save progress: {e}")

            except Exception as row_e:
                logger.error(f"Row {display_idx}: Unexpected error: {row_e}", exc_info=True)
                print(f"ERROR: Row {display_idx}: Unexpected error: {row_e}")

        # Final summary
        chat_generated_questions = df["chat_generated_questions"]
        total_generated = len([x for x in chat_generated_questions if x])
        logger.info("----- Summary -----")
        logger.info(f"Total original records: {len(df)}")
        logger.info(f"Non-null ground_truth_answer used: {gt_series.dropna().shape[0]}")
        logger.info(f"Generated question sets (LLM): {total_generated}")
        logger.info(f"Final results saved to: {output_file}")
        print("INFO: ----- Summary -----")
        print(f"INFO: Total original records: {len(df)}")
        print(f"INFO: Non-null ground_truth_answer used: {gt_series.dropna().shape[0]}")
        print(f"INFO: Generated question sets (LLM): {total_generated}")
        print(f"INFO: Final results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error processing the Excel file: {e}", exc_info=True)
        print(f"ERROR: Error processing the Excel file: {e}")

# Main entry point
if __name__ == "__main__":
    file_path = r"outputs/data_preparation_output_samples/data_prep_1.xlsx"
    output_folder = "outputs/data_preparation_output_samples/"
    logger.info("Launching async processing...")
    print("INFO: Launching async processing...")
    asyncio.run(process_excel(file_path, output_folder))
    logger.info("Processing completed.")
    print("INFO: Processing completed.")
