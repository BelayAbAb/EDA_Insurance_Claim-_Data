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
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report\visualization'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Distribution of Total Premiums by Region
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='Province', y='TotalPremium', palette='viridis')
plt.title('Total Premiums by Region')
plt.xlabel('Region')
plt.ylabel('Total Premium')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_premiums_by_region.jpg'), format='jpg')
plt.close()

# 2. Relationship Between Total Premiums and Total Claims
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='TotalPremium', y='TotalClaims', hue='Province', palette='viridis', alpha=0.7)
sns.regplot(data=data, x='TotalPremium', y='TotalClaims', scatter=False, color='black')
plt.title('Relationship Between Total Premiums and Total Claims')
plt.xlabel('Total Premium')
plt.ylabel('Total Claims')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'premiums_vs_claims.jpg'), format='jpg')
plt.close()

# 3. Box Plot of Total Premiums
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='TotalPremium', palette='viridis')
plt.title('Box Plot of Total Premiums')
plt.xlabel('Total Premium')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'box_plot_total_premiums.jpg'), format='jpg')
plt.close()

print(f"Visualization completed. JPG files saved to: {output_dir}")
