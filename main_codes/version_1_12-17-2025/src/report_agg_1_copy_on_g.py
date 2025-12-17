
import pandas as pd
import os

# ============================================================
# PASS / FAIL CONFIG 
# ============================================================

PASS_FAIL_THRESHOLDS = {
    "groundtruth_accuracy_prompt": 0.75,
    "completeness_evaluation_prompt": 0.70,
    "context_precision_prompt": 0.75,
    "context_recall_prompt": 0.75,
    # thresholds stay the same, but the comparison direction is changed below
    "hallucination_cove_prompt": 0.20,
    "hallucination_consistency_prompt": 0.20,
}

# Direction rules: >= for normal metrics, < for hallucination metrics (lower is better)
PASS_FAIL_RULES = {
    "groundtruth_accuracy_prompt": ">=",
    "completeness_evaluation_prompt": ">=",
    "context_precision_prompt": ">=",
    "context_recall_prompt": ">=",
    # ✅ lower score means pass for hallucination metrics
    "hallucination_cove_prompt": "<",
    "hallucination_consistency_prompt": "<",
}

def is_pass(metric_key: str, score: float) -> bool:
    """
    Returns True if the score passes the threshold for the given metric_key.
    For most metrics: score >= threshold (higher is better).
    For hallucination metrics: score < threshold (lower is better).
    """
    thresh = PASS_FAIL_THRESHOLDS.get(metric_key)
    rule = PASS_FAIL_RULES.get(metric_key, ">=")
    if thresh is None or not (0.0 <= score <= 1.0):
        return False
    if rule == ">=":
        return score >= thresh
    elif rule == "<":
        return score < thresh
    else:
        return False

def pass_fail(metric_key: str, score: float) -> str:
    return "pass" if is_pass(metric_key, score) else "fail"


# Metric mapping DF column → prompt key
# ✅ Fixed mapping for hallucination metrics
METRIC_MAP = {
    "Groundtruth Accuracy": "groundtruth_accuracy_prompt",
    "Completeness": "completeness_evaluation_prompt",
    "Context Precision": "context_precision_prompt",
    "Context Recall": "context_recall_prompt",
    "Hallucination Consistency Score": "hallucination_consistency_prompt",
    "Hallucination CoVe": "hallucination_cove_prompt",
}

# ============================================================
# 1. LOAD EXCEL
# ============================================================

path = r"outputs/metric_evaluation_outputs/metric_evaluation_results_2.xlsx"
xls = pd.ExcelFile(path)
first_sheet = xls.sheet_names[0]
df = pd.read_excel(path, sheet_name=first_sheet)

print("Using sheet:", first_sheet)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 2. TOTAL ROWS
# ============================================================

total_rows = len(df)
print("Total rows:", total_rows)


# ============================================================
# 3. AGGREGATED SCORES (mean values)
# ============================================================

metrics = [
    "Groundtruth Accuracy",
    "Completeness",
    "Context Precision",
    "Context Recall",
    "Hallucination Consistency Score",
    "Hallucination CoVe"
]

agg = pd.DataFrame()
agg.loc[0, 0] = "Aggregated Scores"

# metric names
for i, m in enumerate(metrics):
    agg.loc[0, i+1] = m

# metric mean values
agg.loc[1, 0] = ""
for i, m in enumerate(metrics):
    agg.loc[1, i+1] = round(df[m].mean(), 1)


# ============================================================
# 4. REAL PASS/FAIL FOR AUGMENTED DATA
# ============================================================

aug_types = []
aug_counts = []
aug_pass = []
aug_fail = []

for qtype, group in df.groupby("question_type"):
    aug_types.append(qtype)
    total = len(group)
    aug_counts.append(total)

    # PASS/FAIL using Groundtruth Accuracy
    metric_key = METRIC_MAP["Groundtruth Accuracy"]
    col_name = "Groundtruth Accuracy"

    if col_name in group.columns:
        pass_count = sum(is_pass(metric_key, score) for score in group[col_name])
    else:
        pass_count = 0  # fallback if column missing
    fail_count = total - pass_count

    aug_pass.append(pass_count)
    aug_fail.append(fail_count)

# Build 4-row output block
aug_df = pd.DataFrame()
aug_df.loc[0, 0] = "Disaggregated Analysis with Augmented Data"
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
# 5. REAL PASS/FAIL FOR TOPIC + SUBTOPIC
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

    metric_key = METRIC_MAP["Groundtruth Accuracy"]

    pass_count = sum(is_pass(metric_key, score) for score in group["Groundtruth Accuracy"])
    fail_count = total - pass_count

    topic_pass.append(pass_count)
    topic_fail.append(fail_count)

topic_df = pd.DataFrame()
topic_df.loc[0, 0] = "Disaggregated Analysis with Topic & Sub topics"
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

output_path = os.path.join("reports", "agg_run_1.xlsx")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final.to_excel(output_path, index=False, header=False)

from openpyxl import load_workbook  # add import
wb = load_workbook(output_path)     # define wb

wb.save(output_path)
print("✅ Saved:", output_path)
