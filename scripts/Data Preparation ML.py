import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os

# Define the path to the text file
file_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3.txt'

# Load data from the text file
data = pd.read_csv(file_path, delimiter='|', low_memory=False)

# Display the first few rows of the DataFrame for inspection
print("Initial Data:")
print(data.head())

# Replace empty strings with NaN
data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Define numeric and categorical columns
numeric_columns = [
    'TotalPremium', 'TotalClaims', 'Cylinders', 'cubiccapacity',
    'kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding'
]
categorical_columns = ['Gender', 'VehicleType', 'Province', 'PostalCode']

# Convert numeric columns to numeric types and fill NaN values
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set errors to NaN
    data[col].fillna(data[col].mean(), inplace=True)  # Fill NaN with mean

# Apply OneHotEncoding to categorical columns
encoder = OneHotEncoder(sparse=True, drop='first')  # Use sparse matrix representation
X_encoded = encoder.fit_transform(data[categorical_columns])

# Create a DataFrame from the encoded features (only if needed for debugging)
# encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the numeric features and the encoded categorical features
# Note: If X_encoded is sparse, we can concatenate it with sparse representation directly
from scipy.sparse import hstack
X_numeric = data[numeric_columns].values
X_final = hstack([X_numeric, X_encoded])

# Feature Engineering
data['VehicleAge'] = 2024 - pd.to_datetime(data['RegistrationYear'], format='%Y').dt.year
data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], errors='coerce', format='%d/%m/%Y')
data['PolicyDuration'] = (data['TransactionMonth'] - data['VehicleIntroDate']).dt.days.fillna(0) / 365
data.drop(columns=['TransactionMonth', 'VehicleIntroDate'], inplace=True)

# Prepare features and target variable
X_final = hstack([X_numeric, X_encoded])  # Ensure X_final is updated with new feature engineering
y = data['TotalClaims']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Save the processed data
processed_data_path = r'C:\Users\User\Desktop\10Acadamy\Week-3\Data\MachineLearningRating_v3\processed_data'
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)

np.save(os.path.join(processed_data_path, 'X_train.npy'), X_train)
np.save(os.path.join(processed_data_path, 'X_test.npy'), X_test)
np.save(os.path.join(processed_data_path, 'y_train.npy'), y_train)
np.save(os.path.join(processed_data_path, 'y_test.npy'), y_test)

print(f"Data saved to: {processed_data_path}")

# Optionally, save the final DataFrame to a new CSV file
# X_final_df = pd.DataFrame(X_final.toarray())  # Convert sparse matrix to DataFrame if needed
# X_final_df.to_csv(os.path.join(processed_data_path, 'cleaned_encoded_data.csv'), index=False)
# print("Final Encoded Data saved to: cleaned_encoded_data.csv")