import pandas as pd
import plotly.express as px
from pathlib import Path
# Read data from Excel file
df = pd.read_excel("outputs/augmentations_output_samples/final_augmented_questions_11.xlsx")

# Create the sunburst chart
fig = px.sunburst(
    df,
    path=['topic', 'sub_topic', 'section'],  # Hierarchy
    title="Click any topic → subtopic → section to drill down",
    color='topic',  # Color by top-level topic (consistent colors)
    color_discrete_sequence=px.colors.qualitative.Bold
)

# Update layout for better margins
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

# Combine the relevant columns into a single DataFrame for the table
table_df = df[['topic', 'sub_topic', 'section']]

# Convert the combined table to HTML
table_html = table_df.to_html(index=False, header=True, border=1)

# Create the final HTML structure with the chart and table
html_content = f"""
<html>
<head>
    <meta charset="UTF-8">
    <title>Sunburst Chart and Data Table</title>
</head>
<body>
    <h1>Sunburst Chart</h1>
    <h1>Topic, Sub-Topic, and Section Table</h1>
    {table_html}

    <!-- Embed the Plotly chart directly -->
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}

</body>
</html>
"""

# Save everything in a single HTML file (UTF-8 to avoid UnicodeEncodeError)

# Target file path
output_path = Path("reports/data_coverage_sunburst_chart_samples.html")

# Ensure the folder exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save with UTF-8 encoding
output_path.write_text(html_content, encoding="utf-8")


print("Plotly Sunburst chart and table saved as sunburst_chart_with_table_11.html")

