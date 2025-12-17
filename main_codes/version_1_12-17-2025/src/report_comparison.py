import pandas as pd
import plotly.express as px

# Load the data from the Excel files
file1 = r"reports\agg_score_all_metrics_run_1.xlsx"
file2 = r"reports\agg_score_all_metrics_run_2.xlsx"
df1 = pd.read_excel(file1, sheet_name="Sheet1")
df2 = pd.read_excel(file2, sheet_name="Sheet1")

# ============================================================
# 1. AGGREGATED SCORES (ROW 0 in first 6 columns)
# ============================================================
agg_cols = [
    "Groundtruth Accuracy",
    "Completeness",
    "Context Precision",
    "Context Recall",
    "Hallucination Consistency Score",
    "Hallucination CoVe"
]

agg1 = df1.loc[0, agg_cols]
agg2 = df2.loc[0, agg_cols]

agg_compare = pd.DataFrame({
    "Metric": agg_cols,
    "Run_1": agg1.values.astype(float),
    "Run_2": agg2.values.astype(float)
})

# ============================================================
# 2. DISAGGREGATED AUGMENTED (ROWS 2–5)
# ============================================================
def extract_dis(df):
    names = df.iloc[2, 1:7]
    records = df.iloc[3, 1:7].str.replace(" Records", "").astype(int)
    passed = df.iloc[4, 1:7].str.replace(" Pass", "").astype(int)
    failed = records - passed
    return pd.DataFrame({
        "Metric": names.values,
        "Pass": passed.values,
        "Fail": failed.values
    })

dis1 = extract_dis(df1)
dis2 = extract_dis(df2)

dis_compare = pd.DataFrame({
    "Metric": dis1["Metric"],
    "Run_1 Pass": dis1["Pass"],
    "Run_1 Fail": dis1["Fail"],
    "Run_2 Pass": dis2["Pass"],
    "Run_2 Fail": dis2["Fail"]
})

# ============================================================
# 3. TOPIC & SUBTOPIC (ROWS 7–10)
# ============================================================
def extract_topics(df):
    names = df.iloc[7, 1:].dropna()
    records = df.iloc[8, 1:len(names)+1].str.replace(" Records", "").astype(int)
    passed = df.iloc[9, 1:len(names)+1].str.replace(" Pass", "").astype(int)
    failed = records - passed
    return pd.DataFrame({
        "Topic": names.values,
        "Pass": passed.values,
        "Fail": failed.values
    })

topic1 = extract_topics(df1)
topic2 = extract_topics(df2)

topic_compare = pd.DataFrame({
    "Topic": topic1["Topic"],
    "Run_1 Pass": topic1["Pass"],
    "Run_1 Fail": topic1["Fail"],
    "Run_2 Pass": topic2["Pass"],
    "Run_2 Fail": topic2["Fail"]
})

# ============================================================
# 4. EXPORT TO HTML WITH INTERACTIVE CHARTS
# ============================================================
html_file = "reports/comparison_chart_sample.html"
html = "<html><head><title>Comparison Dashboard</title></head><body>"
html += "<h1>Comparison Dashboard</h1>"

# ------------ Aggregated Chart ------------
fig1 = px.bar(
    agg_compare,
    x="Metric",
    y=["Run_1", "Run_2"],
    title="Aggregated Score Comparison",
    barmode="group",
    color_discrete_sequence=["#6baed6", "#9467bd"]
)

# Add pattern to Run_2 bars
fig1.update_traces(
    marker_pattern_shape="/",
    marker_pattern_solidity=0.4,
    selector=dict(name="Run_2")
)

# Update axis titles
fig1.update_layout(
    xaxis_title="Metrics",
    yaxis_title="Score",
)

# Show the figure
chart1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')

html += "<h2>Aggregated Score Comparison Table</h2>"
html += agg_compare.to_html(index=False)
html += "<h2>Aggregated Chart</h2>"
html += chart1_html

# ------------ Disaggregated Chart ------------
dis_compare_long = dis_compare.melt(
    id_vars=["Metric"],  # Keep the Metric column
    value_vars=["Run_1 Pass", "Run_1 Fail", "Run_2 Pass", "Run_2 Fail"],  # The pass/fail columns
    var_name="Pass/Fail",  # New column for pass/fail
    value_name="Score"  # New column for the values
)

fig2 = px.bar(
    dis_compare_long,
    x="Metric",
    y="Score",
    color="Pass/Fail",
    title="Disaggregated Score Augmented Comparison",
    barmode="group",
    color_discrete_map={
        "Run_1 Pass": "#50ff50",
        "Run_1 Fail": "#ff5050",
        "Run_2 Pass": "#50ff50",
        "Run_2 Fail": "#ff5050"
    }
)

# Add pattern to Run_2 bars
fig2.update_traces(
    marker_pattern_shape="/",
    marker_pattern_solidity=0.4,
    selector=dict(legendgroup="Run_2 Pass")
)
fig2.update_traces(
    marker_pattern_shape="/",
    marker_pattern_solidity=0.4,
    selector=dict(legendgroup="Run_2 Fail")
)

chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

html += "<h2>Disaggregated Score Augmented Comparison Table</h2>"
html += dis_compare.to_html(index=False)
html += "<h2>Disaggregated Score Augmented Chart</h2>"
html += chart2_html

# ------------ Topic Chart ------------
topic_compare_long = topic_compare.melt(
    id_vars=["Topic"],  # Keep the Topic column
    value_vars=["Run_1 Pass", "Run_1 Fail", "Run_2 Pass", "Run_2 Fail"],  # The pass/fail columns
    var_name="Pass/Fail",  # New column for pass/fail
    value_name="Score"  # New column for the values
)

# Calculate total topics and create batch ranges
total_topics = len(topic_compare)
batch_size = 5
num_batches = (total_topics + batch_size - 1) // batch_size  # Ceiling division

fig3 = px.bar(
    topic_compare_long,
    x="Topic",
    y="Score",
    color="Pass/Fail",
    title="Topic & Subtopic Comparison",
    barmode="group",
    color_discrete_map={
        "Run_1 Pass": "#50ff50",
        "Run_1 Fail": "#ff5050",
        "Run_2 Pass": "#50ff50",
        "Run_2 Fail": "#ff5050"
    }
)

# Add pattern to Run_2 bars
fig3.update_traces(
    marker_pattern_shape="/",
    marker_pattern_solidity=0.4,
    selector=dict(legendgroup="Run_2 Pass")
)
fig3.update_traces(
    marker_pattern_shape="/",
    marker_pattern_solidity=0.4,
    selector=dict(legendgroup="Run_2 Fail")
)

# Create buttons for batch filtering
buttons = []

# Add "Show All" button
buttons.append(
    dict(
        label="Show All",
        method="relayout",
        args=[{"xaxis.range": [-0.5, total_topics - 0.5]}]
    )
)

# Add buttons for each batch
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_topics)
    
    buttons.append(
        dict(
            label=f"Topics {start_idx + 1}-{end_idx}",
            method="relayout",
            args=[{"xaxis.range": [start_idx - 0.5, end_idx - 0.5]}]
        )
    )

# Update layout with dropdown menu
fig3.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )
    ],
    xaxis=dict(range=[-0.5, min(batch_size, total_topics) - 0.5])  # Start with first batch visible
)

chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

html += "<h2>Topic Comparison Table</h2>"
html += topic_compare.to_html(index=False)
html += "<h2>Topic Chart</h2>"
html += chart3_html

# --- Save final HTML -----
html += "</body></html>"

with open(html_file, "w") as f:
    f.write(html)

print("comparison_charts.html created successfully!")