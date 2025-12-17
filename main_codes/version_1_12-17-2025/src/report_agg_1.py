
import pandas as pd
import os

# ============================================================
# PASS / FAIL CONFIG (updated logic per your requirement)
# ============================================================

PASS_FAIL_THRESHOLDS = {
    "groundtruth_accuracy_prompt": 0.75,
    "completeness_evaluation_prompt": 0.70,
    "context_precision_prompt": 0.75,
    "context_recall_prompt": 0.75,
    "hallucination_cove_prompt": 0.21,          # lower is better
    "hallucination_consistency_prompt": 0.21,   # lower is better
}

# Direction rules: >= for normal metrics, < for hallucination metrics
PASS_FAIL_RULES = {
    "groundtruth_accuracy_prompt": ">=",
    "completeness_evaluation_prompt": ">=",
    "context_precision_prompt": ">=",
    "context_recall_prompt": ">=",
    "hallucination_cove_prompt": "<",
    "hallucination_consistency_prompt": "<",
}

def is_pass(metric_key: str, score: float) -> bool:
    """
    Return True if score passes the rule for metric_key.
    """
    thresh = PASS_FAIL_THRESHOLDS.get(metric_key)
    rule = PASS_FAIL_RULES.get(metric_key, ">=")
    if thresh is None:
        return False
    if score is None:
        return False
    try:
        s = float(score)
    except Exception:
        return False
    if rule == ">=":
        return s >= thresh
    elif rule == "<":
        return s < thresh
    return False

# Mapping: DataFrame column name -> metric key
METRIC_MAP = {
    "Groundtruth Accuracy": "groundtruth_accuracy_prompt",
    "Completeness": "completeness_evaluation_prompt",
    "Context Precision": "context_precision_prompt",
    "Context Recall": "context_recall_prompt",
    "Hallucination Consistency Score": "hallucination_consistency_prompt",
    "Hallucination CoVe": "hallucination_cove_prompt",
}

ALL_METRIC_COLUMNS = list(METRIC_MAP.keys())

def row_all_metrics_pass(row: pd.Series) -> bool:
    """
    Return True only if the row passes *all* metrics in METRIC_MAP.
    If any metric fails (or column missing), mark as Fail.
    """
    for col, key in METRIC_MAP.items():
        if col not in row.index:
            return False  # missing column -> conservative fail
        if not is_pass(key, row[col]):
            return False
    return True

# ============================================================
# 1. LOAD EXCEL
# ============================================================

path = r"outputs/metrics_output_samples/merged_file_3g_2.xlsx"
xls = pd.ExcelFile(path)
first_sheet = xls.sheet_names[0]
df = pd.read_excel(path, sheet_name=first_sheet)

print("Using sheet:", first_sheet)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# Basic sanity: ensure required grouping columns exist
required_cols = ["question_type", "topic", "sub_topic"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise KeyError(f"Missing required columns: {missing_required}")

# ============================================================
# 2. TOTAL ROWS
# ============================================================

total_rows = len(df)
print("Total rows:", total_rows)

# ============================================================
# 3. AGGREGATED SCORES (mean values)
# ============================================================

metrics = ALL_METRIC_COLUMNS

agg = pd.DataFrame()
agg.loc[0, 0] = "Aggregated Scores"
for i, m in enumerate(metrics):
    agg.loc[0, i+1] = m

agg.loc[1, 0] = ""
for i, m in enumerate(metrics):
    vals = pd.to_numeric(df[m], errors='coerce')
    agg.loc[1, i+1] = round(vals.mean(), 2)

# ============================================================
# 4. DISAGGREGATED ANALYSIS (ALL METRICS) BY question_type
# ============================================================

aug_types = []
aug_counts = []
aug_pass = []
aug_fail = []

for qtype, group in df.groupby("question_type"):
    aug_types.append(qtype)
    total = len(group)
    aug_counts.append(total)

    # Pass only if ALL metrics pass (AND rule)
    pass_count = sum(row_all_metrics_pass(row) for _, row in group.iterrows())
    fail_count = total - pass_count

    aug_pass.append(pass_count)
    aug_fail.append(fail_count)

aug_df = pd.DataFrame()
aug_df.loc[0, 0] = "Disaggregated Analysis based on ALL metrics"
for i, t in enumerate(aug_types):
    aug_df.loc[0, i+1] = t

aug_df.loc[1, 0] = ""
for i, cnt in enumerate(aug_counts):
    aug_df.loc[1, i+1] = f"{cnt} Records"

aug_df.loc[2, 0] = ""
for i, cnt in enumerate(aug_pass):
    aug_df.loc[2, i+1] = f"{cnt} Pass"

aug_df.loc[3, 0] = ""
for i, cnt in enumerate(aug_fail):
    aug_df.loc[3, i+1] = f"{cnt} Fail"

# ============================================================
# 5. DISAGGREGATED ANALYSIS (ALL METRICS) BY Topic & Sub topics
# ============================================================

df["topic_sub"] = df["topic"].astype(str) + " - " + df["sub_topic"].astype(str)

topic_types = []
topic_counts = []
topic_pass = []
topic_fail = []

for ts, group in df.groupby("topic_sub"):
    topic_types.append(ts)
    total = len(group)
    topic_counts.append(total)

    pass_count = sum(row_all_metrics_pass(row) for _, row in group.iterrows())
    fail_count = total - pass_count

    topic_pass.append(pass_count)
    topic_fail.append(fail_count)

topic_df = pd.DataFrame()
topic_df.loc[0, 0] = "Disaggregated Analysis with Topic & Sub topics (ALL metrics)"
for i, t in enumerate(topic_types):
    topic_df.loc[0, i+1] = t

topic_df.loc[1, 0] = ""
for i, cnt in enumerate(topic_counts):
    topic_df.loc[1, i+1] = f"{cnt} Records"

topic_df.loc[2, 0] = ""
for i, cnt in enumerate(topic_pass):
    topic_df.loc[2, i+1] = f"{cnt} Pass"

topic_df.loc[3, 0] = ""
for i, cnt in enumerate(topic_fail):
    topic_df.loc[3, i+1] = f"{cnt} Fail"

# ============================================================
# 6. COMBINE ALL TABLES
# ============================================================

blank = pd.DataFrame([[""]])
final = pd.concat([agg, blank, aug_df, blank, topic_df], ignore_index=True)

# ============================================================
# 7. SAVE OUTPUT FILE
# ============================================================

output_path = os.path.join("reports", "agg_score_all_metrics_run_2.xlsx")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final.to_excel(output_path, index=False, header=False)

from openpyxl import load_workbook
wb = load_workbook(output_path)
wb.save(output_path)
print("✅ Saved:", output_path)
