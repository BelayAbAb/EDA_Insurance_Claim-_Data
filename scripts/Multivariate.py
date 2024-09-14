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

# Define pairs of numeric columns for scatter plots
scatter_pairs = [
    ('TotalPremium', 'TotalClaims'),
    ('Cylinders', 'cubiccapacity'),
    ('CustomValueEstimate', 'CapitalOutstanding'),
    ('TotalPremium', 'CapitalOutstanding'),
    ('NumberOfDoors', 'kilowatts')
]

# Create scatter plots for pairs of numeric columns
fig, axes = plt.subplots(len(scatter_pairs), 1, figsize=(10, 5 * len(scatter_pairs)), constrained_layout=True)

for i, (x_col, y_col) in enumerate(scatter_pairs):
    if x_col in data.columns and y_col in data.columns:
        sns.scatterplot(data=data, x=x_col, y=y_col, ax=axes[i])
        axes[i].set_title(f'Scatter Plot of {x_col} vs {y_col}')
        axes[i].set_xlabel(x_col)
        axes[i].set_ylabel(y_col)
    else:
        axes[i].axis('off')  # Turn off unused subplot

# Save scatter plots as JPG
fig.savefig(os.path.join(output_dir, 'scatter_plots.jpg'), format='jpg')
plt.close(fig)

# Compute and plot correlation matrix
corr_matrix = data[numeric_columns].corr()

# Create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix')

# Save correlation matrix heatmap as JPG
fig.savefig(os.path.join(output_dir, 'correlation_matrix.jpg'), format='jpg')
plt.close(fig)

print(f"Bivariate and multivariate analysis completed. JPG files saved to: {output_dir}")