import os
import pandas as pd

# Define folder path and output file
folder_path = r"C:\Users\T453955\Documents\kaiser_ai_assurance-master_2\merge_data\input_2"
output_file = r"C:\Users\T453955\Documents\kaiser_ai_assurance-master_2\merge_data\merged_file_3g_2.xlsx"

# Initialize an empty DataFrame to store appended data
merged_data = pd.DataFrame()

# Iterate through all Excel files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):  # Check for Excel files
        file_path = os.path.join(folder_path, file_name)
        try:
            # Read the "Main" sheet from the Excel file
            df = pd.read_excel(file_path, sheet_name='Main')
            # Append to the merged DataFrame
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        except Exception as e:
            print(f"Could not read sheet 'Main' from file {file_name}: {e}")

# Save the merged data to a new Excel file with the sheet name "Main"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    merged_data.to_excel(writer, sheet_name='Main', index=False)

print(f"Merged Excel file saved to: {output_file}")