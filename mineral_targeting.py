# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the Excel dataset
file_path = "NGCM-Stream-Sediment-Analysis-Updated.xlsx"  # Correct file name
data = pd.read_excel(file_path)

# Explore the data
print("First 5 rows of the dataset:")
print(data.head())  # Shows the first 5 rows
print("\nColumn names:")
print(data.columns.tolist())  # Lists all column names
print("\nDataset info:")
print(data.info())  # Shows data types and missing values

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())  # Counts missing values in each column

# Select relevant columns (including additional features)
relevant_columns = ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Ni_ppm', 'Zn_ppm', 'Pb_ppm']
data_selected = data[relevant_columns]

# Remove rows with missing values (though none exist here)
data_cleaned = data_selected.dropna()

# Display the cleaned data
print("\nCleaned data (first 5 rows after removing missing values):")
print(data_cleaned.head())
print("\nCleaned data info:")
print(data_cleaned.info())

# Standardize features for better model performance
scaler = StandardScaler()
features = data_cleaned[['X', 'Y', 'Ni_ppm', 'Zn_ppm', 'Pb_ppm']]
scaled_features = scaler.fit_transform(features)
data_cleaned[['X_scaled', 'Y_scaled', 'Ni_ppm_scaled', 'Zn_ppm_scaled', 'Pb_ppm_scaled']] = scaled_features

# Visualize the original data (scatter plots)
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Fe2O3_%'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Iron Oxide (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Iron Oxide Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Au_ppb'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Gold Concentrations')
plt.show()
plt.close()

# Prepare and train model for Copper with optimized features
X = data_cleaned[['X_scaled', 'Y_scaled', 'Ni_ppm_scaled', 'Zn_ppm_scaled', 'Pb_ppm_scaled']]  # Enhanced features
y_cu = data_cleaned['Cu_ppm']

X_train_cu, X_test_cu, y_train_cu, y_test_cu = train_test_split(X, y_cu, test_size=0.2, random_state=42)
model_cu = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # Optimized parameters
model_cu.fit(X_train_cu, y_train_cu)
y_pred_cu = model_cu.predict(X_test_cu)
mse_cu = mean_squared_error(y_test_cu, y_pred_cu)
print("\nMean Squared Error for Copper:", mse_cu)
data_cleaned['Predicted_Cu_ppm'] = model_cu.predict(X)

# Prepare and train model for Iron with optimized features
y_fe = data_cleaned['Fe2O3_%']
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X, y_fe, test_size=0.2, random_state=42)
model_fe = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # Optimized parameters
model_fe.fit(X_train_fe, y_train_fe)
y_pred_fe = model_fe.predict(X_test_fe)
mse_fe = mean_squared_error(y_test_fe, y_pred_fe)
print("Mean Squared Error for Iron:", mse_fe)
data_cleaned['Predicted_Fe2O3_%'] = model_fe.predict(X)

# Prepare and train model for Gold with optimized features
y_au = data_cleaned['Au_ppb']
X_train_au, X_test_au, y_train_au, y_test_au = train_test_split(X, y_au, test_size=0.2, random_state=42)
model_au = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # Optimized parameters
model_au.fit(X_train_au, y_train_au)
y_pred_au = model_au.predict(X_test_au)
mse_au = mean_squared_error(y_test_au, y_pred_au)
print("Mean Squared Error for Gold:", mse_au)
data_cleaned['Predicted_Au_ppb'] = model_au.predict(X)

# Visualize predicted concentrations (scatter plots)
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Fe2O3_%'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Iron Oxide Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Au_ppb'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Gold Concentrations')
plt.show()
plt.close()

# Create heatmaps using 2D histograms
def create_heatmap(x, y, values, title, label):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=values)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label=label)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.show()
    plt.close()

create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Cu_ppm'], 'Heatmap of Predicted Copper Concentrations', 'Predicted Copper (Cu_ppm)')
create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Fe2O3_%'], 'Heatmap of Predicted Iron Oxide Concentrations', 'Predicted Iron Oxide (Fe2O3_%)')
create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Au_ppb'], 'Heatmap of Predicted Gold Concentrations', 'Predicted Gold (Au_ppb)')

# Identify high-potential zones (e.g., top 5% of predicted values)
cu_threshold = data_cleaned['Predicted_Cu_ppm'].quantile(0.95)  # Top 5% for copper
fe_threshold = data_cleaned['Predicted_Fe2O3_%'].quantile(0.95)  # Top 5% for iron
au_threshold = data_cleaned['Predicted_Au_ppb'].quantile(0.95)  # Top 5% for gold
high_potential_cu = data_cleaned[data_cleaned['Predicted_Cu_ppm'] >= cu_threshold]
high_potential_fe = data_cleaned[data_cleaned['Predicted_Fe2O3_%'] >= fe_threshold]
high_potential_au = data_cleaned[data_cleaned['Predicted_Au_ppb'] >= au_threshold]

print("\nHigh-Potential Copper Zones (Top 5%):")
print(high_potential_cu)
print("\nHigh-Potential Iron Zones (Top 5%):")
print(high_potential_fe)
print("\nHigh-Potential Gold Zones (Top 5%):")
print(high_potential_au)

# Save results to a CSV file for submission
output_file = "mineral_targeting_results_enhanced.csv"
data_cleaned.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Visualize high-potential zones (scatter plots)
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential_cu['X'], high_potential_cu['Y'], c='red', label=f'Cu > {cu_threshold:.2f} ppm')
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('High-Potential Copper Zones')
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential_fe['X'], high_potential_fe['Y'], c='blue', label=f'Fe2O3 > {fe_threshold:.2f} %')
plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('High-Potential Iron Zones')
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential_au['X'], high_potential_au['Y'], c='gold', label=f'Au > {au_threshold:.2f} ppb')
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('High-Potential Gold Zones')
plt.legend()
plt.show()
plt.close()

# Generate a summary report
with open("submission_report.txt", "w") as f:
    f.write("Mineral Targeting Analysis Report\n")
    f.write("================================\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    f.write("Methodology:\n")
    f.write("- Used Random Forest Regressor to predict copper, iron, and gold concentrations.\n")
    f.write("- Incorporated additional features: Ni_ppm, Zn_ppm, Pb_ppm, scaled with StandardScaler.\n")
    f.write("- Optimized model with n_estimators=200 and max_depth=10.\n")
    f.write("- Identified high-potential zones using the top 5% of predicted values.\n")
    f.write("- Visualized data with scatter plots and heatmaps.\n\n")
    f.write("Model Performance:\n")
    f.write(f"- Mean Squared Error for Copper: {mse_cu:.2f}\n")
    f.write(f"- Mean Squared Error for Iron: {mse_fe:.2f}\n")
    f.write(f"- Mean Squared Error for Gold: {mse_au:.2f}\n\n")
    f.write("High-Potential Zones:\n")
    f.write(f"- Copper Zones (>{cu_threshold:.2f} ppm): {len(high_potential_cu)} locations\n")
    f.write(f"- Iron Zones (>{fe_threshold:.2f} %): {len(high_potential_fe)} locations\n")
    f.write(f"- Gold Zones (>{au_threshold:.2f} ppb): {len(high_potential_au)} locations\n")
    f.write("\nData saved to: mineral_targeting_results_enhanced.csv")
print("\nSummary report saved to submission_report.txt")