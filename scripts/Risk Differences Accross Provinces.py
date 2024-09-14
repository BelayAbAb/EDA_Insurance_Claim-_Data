import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report\risk_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define risk groups based on TotalClaims (you can adjust this criterion)
threshold = data['TotalClaims'].median()  # Median as a threshold for high and low risk
data['RiskGroup'] = ['High' if x > threshold else 'Low' for x in data['TotalClaims']]

# Segregate control and test groups
group_a = data[data['RiskGroup'] == 'Low']  # Control group
group_b = data[data['RiskGroup'] == 'High']  # Test group

# Statistical Testing: T-tests for KPIs
results = []
for col in numeric_columns:
    if col in group_a.columns and col in group_b.columns:
        t_stat, p_value = stats.ttest_ind(group_a[col].dropna(), group_b[col].dropna(), equal_var=False)
        significance = 'Yes' if p_value < 0.05 else 'No'
        results.append([col, t_stat, p_value, significance])

# Create a 1x3 grid figure with a table for statistical results
fig, axes = plt.subplots(1, 3, figsize=(18, 12), constrained_layout=True)

# Distribution of Total Claims by Risk Group
sns.histplot(group_a['TotalClaims'].dropna(), kde=True, ax=axes[0], color='blue', label='Low Risk')
sns.histplot(group_b['TotalClaims'].dropna(), kde=True, ax=axes[0], color='red', label='High Risk')
axes[0].set_title('Distribution of Total Claims by Risk Group')
axes[0].legend()

# Box Plot of Total Premium by Risk Group
sns.boxplot(data=data, x='RiskGroup', y='TotalPremium', ax=axes[1], palette='viridis')
axes[1].set_title('Box Plot of Total Premium by Risk Group')

# Scatter Plot of Total Premium vs Total Claims
sns.scatterplot(data=data, x='TotalPremium', y='TotalClaims', hue='RiskGroup', palette='viridis', alpha=0.7, ax=axes[2])
axes[2].set_title('Total Premium vs Total Claims by Risk Group')

# Create a table of results
fig_table, ax_table = plt.subplots(figsize=(10, 4))  # Adjust size as needed
ax_table.axis('off')  # Turn off the axis

# Table data
table_data = [['Column', 'T-statistic', 'P-value', 'Significant']] + results
table = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2, 3])

# Save the figures
fig.savefig(os.path.join(output_dir, 'risk_analysis_1x3_grid.jpg'), format='jpg')
fig_table.savefig(os.path.join(output_dir, 'statistical_results_table.jpg'), format='jpg')
plt.close(fig)
plt.close(fig_table)

print(f"Risk analysis and statistical results completed. JPG files saved to: {output_dir}")
