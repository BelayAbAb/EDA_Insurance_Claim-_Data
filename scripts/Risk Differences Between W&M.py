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
numeric_columns = ['TotalClaims']
data = convert_comma_to_dot_and_to_numeric(data, numeric_columns)

# Define output directory for JPG files
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report\Risk Differences Between Women and Men'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Check if 'Gender' column exists
if 'Gender' not in data.columns:
    raise ValueError("'Gender' column not found in the dataset. Please check the column names and update the code accordingly.")

# Segment data into male and female groups
male_risk_data = data[data['Gender'] == 'Male']
female_risk_data = data[data['Gender'] == 'Female']

# Statistical Testing: T-tests for TotalClaims between male and female
results = []
col = 'TotalClaims'
if col in data.columns:
    male_values = male_risk_data[col].dropna()
    female_values = female_risk_data[col].dropna()
    t_stat, p_value = stats.ttest_ind(male_values, female_values, equal_var=False)
    significance = 'Yes' if p_value < 0.05 else 'No'
    results.append([col, t_stat, p_value, significance])

# Create a 1x2 grid figure for visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Distribution of Total Claims by Gender
sns.histplot(male_risk_data['TotalClaims'].dropna(), kde=True, ax=axes[0], color='blue', label='Male')
sns.histplot(female_risk_data['TotalClaims'].dropna(), kde=True, ax=axes[0], color='pink', label='Female')
axes[0].set_title('Distribution of Total Claims by Gender')
axes[0].set_xlabel('Total Claims')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Box Plot of Total Claims by Gender
sns.boxplot(data=data, x='Gender', y='TotalClaims', ax=axes[1], palette='Set2')
axes[1].set_title('Box Plot of Total Claims by Gender')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Total Claims')

# Create a table of results
fig_table, ax_table = plt.subplots(figsize=(8, 3))  # Adjust size as needed
ax_table.axis('off')  # Turn off the axis

# Table data
table_data = [['Column', 'T-statistic', 'P-value', 'Significant']] + results
table = ax_table.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2, 3])

# Save the figures
fig.savefig(os.path.join(output_dir, 'risk_differences_by_gender_1x2_grid.jpg'), format='jpg')
fig_table.savefig(os.path.join(output_dir, 'risk_differences_table.jpg'), format='jpg')
plt.close(fig)
plt.close(fig_table)

print(f"Risk differences between women and men analyzed. JPG files saved to: {output_dir}")
