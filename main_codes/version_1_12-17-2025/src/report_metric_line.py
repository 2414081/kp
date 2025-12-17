import pandas as pd
import plotly.express as px

# Load file
file = r"outputs/metrics_output_samples/merged_file_3g.xlsx"
df = pd.read_excel(file, sheet_name="Main")

metrics = [
    "Groundtruth Accuracy",
    "Completeness",
    "Context Precision",
    "Context Recall",
    "Hallucination Consistency Score",
    "Hallucination CoVe"
]

# Create clean numeric table
clean_df = pd.DataFrame()

for m in metrics:
    clean_df[m] = pd.to_numeric(df[m], errors="coerce")   # convert to numbers
    clean_df[m] = clean_df[m].dropna().reset_index(drop=True)   # keep only numbers

# Align lengths (fill shorter columns with None)
max_len = clean_df.apply(lambda col: col.count()).max()
clean_df = clean_df.reindex(range(max_len))

# ============================================================
# ADDITION: SAVE SAME DATA + INTERACTIVE CHART TO HTML WITH FILTER
# ============================================================
html_file = "reports/metric_line_chart_output.html"

# Add index column
clean_df_html = clean_df.copy()
clean_df_html.insert(0, "Index", range(len(clean_df)))

# Create interactive line chart
# Create interactive line chart with dots (markers)
fig = px.line(
    clean_df_html,
    x="Index",
    y=metrics,
    title="Metric Trend Line Chart (Interactive)"
)

# Update the chart to include both lines and markers (dots)
fig.update_traces(mode='lines+markers')  # Adds dots to the lines

fig.update_layout(
    xaxis_title="Index",  # Set your X-axis title here
    yaxis_title="Score",  # Set your Y-axis title here
)

# Add buttons for the filter (first 10, next 10, etc.)
buttons = []
num_rows = len(clean_df_html)

# Create buttons to show 10 rows at a time
for i in range(0, num_rows, 10):
    end_index = i + 10 if i + 10 < num_rows else num_rows
    label = f"Rows {i+1} to {end_index}"

    # Filter data for this range and create a button
    button = dict(
        label=label,
        method="update",
        args=[{
            "x": [clean_df_html["Index"][i:end_index]],
            "y": [clean_df_html[m][i:end_index] for m in metrics],
            "title": f"Metric Trend (Rows {i+1} to {end_index})"
        }]
    )
    buttons.append(button)

# Add the filter buttons to the layout
fig.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 1.15,
        "y": 1.1
    }]
)

# Convert the figure to HTML
chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# Build final HTML file
html = "<html><head><title>Metric Trend Chart</title></head><body>"
html += "<h1>Metric Trend Line Chart Dashboard</h1>"

html += "<h2>Metric Data Table</h2>"
html += clean_df_html.to_html(index=False)

html += "<h2>Interactive Line Chart with Filter and Dots</h2>"
html += chart_html

html += "</body></html>"

# Save HTML
with open(html_file, "w") as f:
    f.write(html)

print("Saved HTML:", html_file)
