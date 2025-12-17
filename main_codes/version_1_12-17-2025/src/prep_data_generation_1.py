import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from collections import defaultdict
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'configs' and other top-level packages can be imported
PROJECT_ROOT = Path(__file__).resolve( ).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.api_chat import send_question_to_chat_api


# Create logs directory (and subdirectory) if they don't exist
os.makedirs("logs/prep_data_generation_logs", exist_ok=True)

# Configure logging to write to both file and console, with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if this module is reloaded/executed multiple times
if not logger.handlers:
    # File handler with rotation
    file_handler = RotatingFileHandler(
        filename=r"logs/prep_data_generation_logs/prep_data_generation_1_log.log",
        mode="a",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Run started: prep_data_generation_1")

# Load input Excel file containing questions
file_path = r"outputs/augmentations_output_samples/final_augmented_questions_samples_2.xlsx"
logger.info(f"Loading input Excel: {file_path}")
try:
    df = pd.read_excel(file_path)
except FileNotFoundError as e:
    logger.error(f"Input file not found: {file_path} | {e}")
    # Ensure handlers flush before exit
    for h in logger.handlers:
        h.flush()
    raise
except Exception as e:
    logger.error(f"Failed to read Excel '{file_path}': {e}")
    for h in logger.handlers:
        h.flush()
    raise

# Track total number of questions to process
TOTAL_ROWS = len(df)
logger.info(f"Total questions: {TOTAL_ROWS}")
print(f"Total questions: {TOTAL_ROWS}")

# Configure how often to save intermediate results (1 = save after every question)
SAVE_EVERY_N = 1  # set to 1 to save after every question; raise for better performance
BASE_OUTPUT = "outputs/data_preparation_output_samples/data_prep_1.xlsx"
os.makedirs(os.path.dirname(BASE_OUTPUT), exist_ok=True)
logger.info(f"Output will be saved to: {BASE_OUTPUT} every {SAVE_EVERY_N} row(s)")

# Verify the required 'question' column exists in the DataFrame
if "question" in df.columns:

    def process_chat_api_parallel(data, summary_col, snippets_col=None, save_path=None, max_workers=4):
        """
        Parallel processing: 4 concurrent API calls, no duplicate question requests.
        If the same question appears multiple times in the DataFrame, the API is called once,
        and the result is applied to all rows with that question.
        """
        # Ensure columns exist
        if summary_col not in data.columns:
            data[summary_col] = ""
            logger.info(f"Created column: {summary_col}")
        if snippets_col and snippets_col not in data.columns:
            data[snippets_col] = None
            logger.info(f"Created column: {snippets_col}")

        # Build unique question list and index mapping
        index_map = defaultdict(list)  # question -> list of indices
        for idx, q in enumerate(data["question"].tolist()):
            index_map[str(q)].append(idx)
        unique_questions = list(index_map.keys())
        logger.info(f"Parallel mode: unique questions={len(unique_questions)} total_rows={len(data)} workers={max_workers}")

        # Worker
        def worker(question):
            try:
                res = send_question_to_chat_api(question)
                return (question, res, None)
            except Exception as e:
                return (question, None, e)

        # Submit in batches using a thread pool
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(worker, q): q for q in unique_questions}
            for fut in as_completed(futures):
                q = futures[fut]
                question, result, err = fut.result()
                indices = index_map[question]

                if err:
                    for i in indices:
                        logger.warning(f"[CHAT][PAR] FAILED row {i} question='{question}': {err}")
                    # Continue to next future
                else:
                    # Apply to all rows that share the question
                    summarization = result.get("summarization", "")
                    snippet_list = result.get("snippets", [])
                    for i in indices:
                        data.at[i, summary_col] = summarization
                        if snippets_col is not None:
                            data.at[i, snippets_col] = snippet_list
                    # Print the response for this unique question
                    print(f"[CHAT][PAR] question: {question}")
                    print(f"[CHAT][PAR] summary_len={len(summarization)} summary: {summarization}")
                    print(f"[CHAT][PAR] snippets_count={len(snippet_list)} snippets: {snippet_list}")
                    logger.info(f"[CHAT][PAR] OK question_len={len(question)} rows={len(indices)} summary_len={len(summarization)} snippets={len(snippet_list)}")

                completed += len(indices)
                print(f"[CHAT][PAR] progress {completed}/{len(data)}")

                # Incremental save on progress boundaries
                if save_path and (completed % SAVE_EVERY_N == 0):
                    try:
                        data.to_excel(save_path, index=False)
                        logger.info(f"[CHAT][PAR] Incremental save: progress {completed}/{len(data)} -> {save_path}")
                    except Exception as e:
                        logger.error(f"[CHAT][PAR] Failed incremental save at progress {completed}: {e}")
                    finally:
                        for h in logger.handlers:
                            h.flush()
        return data

    # Process each question 3 times through Chat API only
    for i in range(1, 4):
        chat_col = f"actual_chat_response_{i}"
        chat_snippets_col = "retrieval_context_chat" if i == 1 else None

        logger.info(f"Starting parallel chat pass {i} -> summary_col={chat_col} snippets_col={chat_snippets_col}")
        # Run parallel processing with 4 workers
        df = process_chat_api_parallel(df, chat_col, chat_snippets_col, save_path=BASE_OUTPUT, max_workers=4)
        try:
            df.to_excel(BASE_OUTPUT, index=False)
            logger.info(f"Finished parallel chat pass {i} (saved to {BASE_OUTPUT})")
        except Exception as e:
            logger.error(f"Failed to save after parallel chat pass {i}: {e}")
        finally:
            for h in logger.handlers:
                h.flush()

    # Final save after all iterations complete
    try:
        df.to_excel(BASE_OUTPUT, index=False)
        logger.info(f"Final cumulative save: {BASE_OUTPUT}")
    except Exception as e:
        logger.error(f"Final save failed: {e}")
    finally:
        logger.info("Run finished: prep_data_generation_1")
        for h in logger.handlers:
            h.flush()
else:
    # Error if 'question' column is missing from input file
    logger.error("Column 'question' missing.")
    logger.info("Run finished with error: missing 'question' column")
    for h in logger.handlers:
        h.flush()