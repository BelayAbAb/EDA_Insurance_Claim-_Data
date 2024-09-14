import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the path to the text file
file_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3.txt'

# Check if the file exists at the specified path
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist. Please check the file path and name.")

# Load data from the text file
data = pd.read_csv(file_path, delimiter='|')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Data Quality Assessment

# Check for missing values
missing_values = data.isnull().sum()

# Display missing values summary
print("\nMissing Values Summary:")
print(missing_values)

# Handle missing values
# Option 1: Remove rows with missing values
data_cleaned = data.dropna()

# Option 2: Impute missing values (for demonstration, we use mean imputation for numerical columns)
data_imputed = data.fillna({
    'TotalPremium': data['TotalPremium'].mean(),
    'TotalClaims': data['TotalClaims'].mean()
})

# Define output directory and ensure it exists
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the cleaned and imputed data
cleaned_file_path = os.path.join(output_dir, 'insurance_data_cleaned.txt')
imputed_file_path = os.path.join(output_dir, 'insurance_data_imputed.txt')

data_cleaned.to_csv(cleaned_file_path, sep='|', index=False)
data_imputed.to_csv(imputed_file_path, sep='|', index=False)

# Descriptive Statistics
descriptive_stats = data[['TotalPremium', 'TotalClaims']].describe()
print("\nDescriptive Statistics for 'TotalPremium' and 'TotalClaims':")
print(descriptive_stats)

# Calculate and display variability (standard deviation)
std_dev = data[['TotalPremium', 'TotalClaims']].std()
print("\nStandard Deviation for 'TotalPremium' and 'TotalClaims':")
print(std_dev)

# Visualizations
plt.figure(figsize=(14, 6))

# Histogram for TotalPremium
plt.subplot(1, 2, 1)
sns.histplot(data['TotalPremium'], bins=30, kde=True)
plt.title('Distribution of TotalPremium')

# Histogram for TotalClaims
plt.subplot(1, 2, 2)
sns.histplot(data['TotalClaims'], bins=30, kde=True)
plt.title('Distribution of TotalClaims')

# Save plots as JPG files
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_premium_distribution.jpg'), format='jpg')
plt.savefig(os.path.join(output_dir, 'total_claims_distribution.jpg'), format='jpg')

# Show plots (optional, can be commented out if not needed)
plt.show()
