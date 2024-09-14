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
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Select 9 columns for plotting
plot_columns = [
    'TotalPremium', 'TotalClaims',  # Numeric
    'VehicleType', 'make',          # Categorical
    'Gender', 'Province',           # Categorical
    'RegistrationYear', 'CustomValueEstimate', 'CapitalOutstanding'  # Numeric
]

# Create a 3x3 grid figure
fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot histograms for selected columns
for i, col in enumerate(plot_columns):
    if col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'bool':
            # For categorical columns, use a count plot
            sns.countplot(data=data, x=col, ax=axes[i])
            axes[i].set_title(f'Bar Chart of {col}')
            axes[i].tick_params(axis='x', rotation=45)  # Rotate x labels if needed
        else:
            # For numeric columns, use a histogram
            sns.histplot(data[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
    else:
        axes[i].axis('off')  # Turn off unused subplot

# Save the grid figure as JPG
fig.savefig(os.path.join(output_dir, 'univariate_analysis_3x3_grid.jpg'), format='jpg')
plt.close(fig)

print(f"Univariate analysis completed. JPG file saved to: {output_dir}")
