import pandas as pd
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

# Data Structure Review

# Display data types of each column
print("\nData Types for Each Column:")
print(data.dtypes)

# Check for date columns and their format
date_columns = ['TransactionMonth']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')  # Convert to datetime, invalid parsing will be set as NaT
        print(f"\n{col} column formatted as datetime:")
        print(data[col].head())

# Check for categorical columns and their unique values
categorical_columns = ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 'make', 'Model', 'BodyType', 'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType']
for col in categorical_columns:
    if col in data.columns:
        print(f"\nUnique values in the {col} column:")
        print(data[col].unique())

# Check for numerical columns and their ranges
numerical_columns = ['TotalPremium', 'TotalClaims', 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding', 'ExcessSelected', 'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm']
for col in numerical_columns:
    if col in data.columns:
        print(f"\nSummary statistics for {col}:")
        print(data[col].describe())

# Save cleaned data (with corrected data types)
output_dir = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cleaned_file_path = os.path.join(output_dir, 'insurance_data_structured_review.txt')
data.to_csv(cleaned_file_path, sep='|', index=False)

print("\nData structure review completed. Cleaned data saved to:")
print(cleaned_file_path)