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
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report\data_comparison'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Analyze trends across geographic regions
# Aggregate data by region (assuming 'Province' is the geographic region column)
region_data = data.groupby('Province').agg({
    'TotalPremium': 'sum',
    'TotalClaims': 'sum',
    'CustomValueEstimate': 'mean',
    'CapitalOutstanding': 'mean'
}).reset_index()

# Create bar plots for total premiums and claims by region
fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

# Total Premiums by Region
sns.barplot(data=region_data, x='Province', y='TotalPremium', ax=axes[0], palette='viridis')
axes[0].set_title('Total Premiums by Region')
axes[0].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

# Total Claims by Region
sns.barplot(data=region_data, x='Province', y='TotalClaims', ax=axes[1], palette='viridis')
axes[1].set_title('Total Claims by Region')
axes[1].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

# Save the bar plots as JPG
fig.savefig(os.path.join(output_dir, 'total_premiums_and_claims_by_region.jpg'), format='jpg')
plt.close(fig)

# Create a line plot for average custom value estimate and capital outstanding by region
fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

# Average Custom Value Estimate by Region
sns.lineplot(data=region_data, x='Province', y='CustomValueEstimate', marker='o', ax=axes[0], palette='viridis')
axes[0].set_title('Average Custom Value Estimate by Region')
axes[0].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

# Average Capital Outstanding by Region
sns.lineplot(data=region_data, x='Province', y='CapitalOutstanding', marker='o', ax=axes[1], palette='viridis')
axes[1].set_title('Average Capital Outstanding by Region')
axes[1].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability

# Save the line plots as JPG
fig.savefig(os.path.join(output_dir, 'avg_custom_value_estimate_and_capital_outstanding_by_region.jpg'), format='jpg')
plt.close(fig)

print(f"Data comparison across geographic regions completed. JPG files saved to: {output_dir}")

