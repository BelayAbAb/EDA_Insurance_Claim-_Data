import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the path to the text file
file_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3.txt'

# Load data from the text file with specified dtype options
data = pd.read_csv(file_path, delimiter='|', low_memory=False)

# Replace commas with periods in numeric columns and convert to numeric
def convert_comma_to_dot_and_to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            # Replace commas with periods
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            # Convert to numeric, forcing errors to NaN (use 'coerce' mode)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# List of columns to convert to numeric
numeric_columns = ['TotalPremium', 'TotalClaims', 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding']
data = convert_comma_to_dot_and_to_numeric(data, numeric_columns)

# Define output directory for JPG files
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report\outlier_detection'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Number of numeric columns
num_columns = len(numeric_columns)

# Create a 3x3 grid figure
fig, axes = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot box plots for each numeric column
for i in range(9):
    if i < num_columns:
        col = numeric_columns[i]
        sns.boxplot(data=data, x=col, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    else:
        axes[i].axis('off')  # Turn off unused subplot

# Save the grid of box plots as JPG
fig.savefig(os.path.join(output_dir, 'box_plots_3x3_grid.jpg'), format='jpg')
plt.close(fig)

print(f"Outlier detection completed. JPG file saved to: {output_dir}")