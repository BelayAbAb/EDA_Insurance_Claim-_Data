import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the correct path to the text file
file_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3.txt'

# Check if the file exists at the specified path
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist. Please check the file path and name.")

# Load data from the text file
data = pd.read_csv(file_path, delimiter='|')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Calculate descriptive statistics for numerical features
descriptive_stats = data[['TotalPremium', 'TotalClaims']].describe()

# Display descriptive statistics
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
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Report'
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_premium_distribution.jpg'), format='jpg')
plt.savefig(os.path.join(output_dir, 'total_claims_distribution.jpg'), format='jpg')

# Show plots (optional, can be commented out if not needed)
plt.show()
