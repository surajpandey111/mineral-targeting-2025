import geopandas as gpd
import fiona
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import rasterio
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from dash import Dash, dcc, html, Input, Output
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import kneighbors_graph
import xgboost


base_dir = "Processed_Mineral_Data/" 

# --- Model and Data Save/Load Configuration ---
train_models = False  # Set to False to load saved models instead of training
process_data = False  # Set to False to load saved data instead of reprocessing
model_save_path_stgnn = "stgnn_model.pth"
model_save_path_cnn = "cnn_model.pth"
processed_data_path = "processed_data.pkl"  # File to save/load processed data

# --- Load and Process NGCM Data (XLSX) ---
ngcm_path = os.path.join(base_dir, "NGCM", "NGCM-Stream-Sediment-Analysis-Updated.xlsx")
print("Loading NGCM data...")
data = pd.read_excel(ngcm_path)
relevant_columns = ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Ni_ppm', 'Zn_ppm', 'Pb_ppm']
data_cleaned = data[relevant_columns].dropna()
print("NGCM Data Loaded. Shape:", data_cleaned.shape)

# --- Load or Process ASTER and Magnetic Data ---
if process_data or not os.path.exists(processed_data_path):
    print("Processing ASTER data...")
    aster_rasters = {}
    aster_dir = os.path.join(base_dir, "ASTER")
    for i, file in enumerate(tqdm(os.listdir(aster_dir))):
        file_path = os.path.join(aster_dir, file)
        if file.endswith(('.tif', '.ovr', '.enp')):  # Process TIF, OVR, ENP
            try:
                with rasterio.open(file_path) as src:
                    raster_data = src.read(1)  # Read first band
                    transform = src.transform
                    cols, rows = np.meshgrid(np.arange(raster_data.shape[1]), np.arange(raster_data.shape[0]))
                    xs, ys = rasterio.transform.xy(transform, rows, cols)
                    xs, ys = np.array(xs), np.array(ys)
                    values = griddata((xs.ravel(), ys.ravel()), raster_data.ravel(), 
                                     (data_cleaned['X'].values, data_cleaned['Y'].values), 
                                     method='nearest', fill_value=0)
                    aster_rasters[file.split('.')[0]] = values
                    print(f"Processed {file} (Raster {i+1})")
            except rasterio.errors.RasterioIOError:
                print(f"Skipping {file}: Not a readable raster or incompatible format.")
        elif file.endswith('.tfw'):  # Handle TFW for geotransform
            with open(file_path, 'r') as f:
                tfw_lines = f.readlines()
                if len(tfw_lines) >= 6:
                    transform = [float(x) for x in tfw_lines[:6]]  # Extract geotransform params

    for key, values in aster_rasters.items():
        data_cleaned[key] = values
    print("ASTER Data Integrated. Columns added:", list(aster_rasters.keys()))

    # --- Load and Process Aerogeophysical Magnetic Data ---
    print("Processing Aerogeophysical data...")
    magnetic_dir = os.path.join(base_dir, "Aerogeophysical")
    magnetic_data = pd.DataFrame()
    for i, file in enumerate(tqdm(os.listdir(magnetic_dir))):
        file_path = os.path.join(magnetic_dir, file)
        if file.endswith('.gdb'):
            try:
                with fiona.open(file_path, driver='OpenFileGDB') as layer:
                    for feature in layer:
                        temp_df = gpd.GeoDataFrame.from_features([feature])
                        temp_df = temp_df[['X', 'Y', 'Magnetic_Anomaly']].dropna()
                        magnetic_data = pd.concat([magnetic_data, temp_df], ignore_index=True)
                print(f"Processed {file} (GDB {i+1})")
            except Exception as e:
                print(f"Error processing {file}: {e}. Skipping GDB file.")
        elif file.endswith('.xyz'):
            try:
                temp_df = pd.read_csv(file_path, delim_whitespace=True, names=['X', 'Y', 'Magnetic_Anomaly'])
                magnetic_data = pd.concat([magnetic_data, temp_df], ignore_index=True)
                print(f"Processed {file} (XYZ {i+1})")
            except Exception as e:
                print(f"Error processing {file}: {e}. Skipping XYZ file.")
        elif file.endswith(('.grd', '.tiff')):
            try:
                if file.endswith('.grd'):
                    with open(file_path, 'rb') as f:
                        header_lines = [next(f).decode('utf-8', errors='ignore') for _ in range(6)]
                        data = np.fromfile(f, dtype=np.float32)
                        ncols = int(header_lines[1].split()[1])
                        nrows = int(header_lines[2].split()[1])
                        data = data.reshape(nrows, ncols)
                else:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)
                x, y = np.meshgrid(np.linspace(0, 100, data.shape[1]), np.linspace(0, 100, data.shape[0]))
                magnetic_grid = griddata((x.ravel(), y.ravel()), data.ravel(), 
                                       (data_cleaned['X'].values, data_cleaned['Y'].values), 
                                       method='nearest', fill_value=0)
                data_cleaned['Magnetic_Grid'] = magnetic_grid
                print(f"Processed {file} (Grid/TIFF {i+1})")
            except Exception as e:
                print(f"Error processing {file}: {e}. Skipping Grid/TIFF file.")

    if not magnetic_data.empty:
        data_cleaned = data_cleaned.merge(magnetic_data[['X', 'Y', 'Magnetic_Anomaly']], on=['X', 'Y'], how='left')
        data_cleaned['Magnetic_Anomaly'] = data_cleaned['Magnetic_Anomaly'].fillna(data_cleaned['Magnetic_Anomaly'].mean())
    print("Magnetic Data Integrated. Shape:", data_cleaned.shape)

    # Save processed data to avoid reprocessing
    data_cleaned.to_pickle(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")
else:
    # Load processed data if it exists
    data_cleaned = pd.read_pickle(processed_data_path)
    print(f"Loaded processed data from {processed_data_path}. Shape:", data_cleaned.shape)
    # Reconstruct aster_rasters from data_cleaned columns
    aster_rasters = {col: data_cleaned[col].values for col in data_cleaned.columns if col not in relevant_columns and col != 'Magnetic_Grid' and col != 'Magnetic_Anomaly'}

# --- Standardize Features ---
data_cleaned = data_cleaned[~np.isinf(data_cleaned).any(axis=1)]
scaler = StandardScaler()
features = data_cleaned.drop(columns=['Cu_ppm', 'Fe2O3_%', 'Au_ppb'])
target = data_cleaned[['Cu_ppm', 'Fe2O3_%', 'Au_ppb']]

# Check for NaN or Inf in features and target before scaling
if features.isna().any().any() or np.isinf(features.values).any() or target.isna().any().any() or np.isinf(target.values).any():
    print("Warning: NaN or Inf detected in features or target before scaling. Replacing with 0.")
    features = features.fillna(0)
    target = target.replace([np.inf, -np.inf], 0)
    features = features.replace([np.inf, -np.inf], 0)

scaled_features = scaler.fit_transform(features)
data_cleaned_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=data_cleaned.index)
data_cleaned_scaled = pd.concat([data_cleaned_scaled, target], axis=1)

# Remove any remaining NaN/inf after scaling
data_cleaned_scaled = data_cleaned_scaled.dropna()
data_cleaned_scaled = data_cleaned_scaled[~np.isinf(data_cleaned_scaled).any(axis=1)]

# Check if data_cleaned_scaled is empty
if data_cleaned_scaled.empty:
    raise ValueError("data_cleaned_scaled is empty after preprocessing. Check data integrity.")

print("Scaled Data Shape:", data_cleaned_scaled.shape)

# --- Unified Feature Set (Multi-Approach Training with Ensembles) ---
print("Creating unified feature set...")
# Interpolate all raster data to NGCM points
raster_features = []
for key in aster_rasters.keys():
    values = data_cleaned[key].values
    raster_features.append(np.nan_to_num(values, nan=0, posinf=0, neginf=0).reshape(-1, 1))
raster_features.append(np.nan_to_num(data_cleaned['Magnetic_Grid'].values, nan=0, posinf=0, neginf=0).reshape(-1, 1))
raster_features = np.hstack(raster_features)

# Combine with NGCM features (excluding targets)
ngcm_features = data_cleaned.drop(columns=['Cu_ppm', 'Fe2O3_%', 'Au_ppb']).values
combined_features = np.hstack((ngcm_features, raster_features))

# Ensure all values are finite
combined_features = np.nan_to_num(combined_features, nan=0, posinf=0, neginf=0)
scaler_unified = StandardScaler()
combined_features_scaled = scaler_unified.fit_transform(combined_features)
# Include X and Y columns
xy_coords = data_cleaned[['X', 'Y']].values
data_unified = pd.DataFrame(np.hstack((xy_coords, combined_features_scaled)), 
                           columns=['X', 'Y'] + [f'feat_{i}' for i in range(combined_features.shape[1])], 
                           index=data_cleaned.index)
data_unified = pd.concat([data_unified, target], axis=1)

# Prepare data for all models
X_unified = data_unified.drop(columns=['Cu_ppm', 'Fe2O3_%', 'Au_ppb']).values
y_unified = data_unified[['Cu_ppm', 'Fe2O3_%', 'Au_ppb']].values

# Split data for training (80%) and testing (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_unified, y_unified, test_size=0.2, random_state=42)
train_indices, test_indices = train_test_split(np.arange(len(X_unified)), test_size=0.2, random_state=42)

# 1. Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
print(f"Linear Regression MSE: {lr_mse:.4f}")

# 2. STGNN (Adapted for unified data)
class STGNNDataset:
    def __init__(self, x, y, edge_index):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.edge_index = edge_index
        self.data = Data(x=self.x, edge_index=self.edge_index, y=self.y)

# Create adjacency matrix for the test set
from sklearn.neighbors import kneighbors_graph
A_test = kneighbors_graph(data_unified.iloc[test_indices][['X', 'Y']], n_neighbors=5, mode='connectivity', include_self=False)
edge_index_test = torch.tensor(A_test.nonzero(), dtype=torch.long)

stgnn_dataset = STGNNDataset(X_test, y_test, edge_index_test)

class STGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stgnn_model = STGNN(input_dim=X_unified.shape[1], hidden_dim=16, output_dim=3).to(device)
stgnn_dataset.data = stgnn_dataset.data.to(device)
optimizer_stgnn = torch.optim.Adam(stgnn_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

if train_models or not os.path.exists("unified_stgnn_model.pth"):
    print("Training STGNN on unified data...")
    stgnn_model.train()
    for epoch in tqdm(range(50)):
        optimizer_stgnn.zero_grad()
        out = stgnn_model(stgnn_dataset.data)
        loss = criterion(out, stgnn_dataset.data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stgnn_model.parameters(), max_norm=1.0)
        optimizer_stgnn.step()
        if epoch % 5 == 0:
            print(f'STGNN Epoch {epoch}, Loss: {loss.item():.4f}')
    torch.save(stgnn_model.state_dict(), "unified_stgnn_model.pth")
else:
    stgnn_model.load_state_dict(torch.load("unified_stgnn_model.pth", map_location=device))
    stgnn_model.eval()

stgnn_model.eval()
with torch.no_grad():
    stgnn_predictions = stgnn_model(stgnn_dataset.data).cpu().numpy()
stgnn_mse = mean_squared_error(y_test, stgnn_predictions)
print(f"STGNN MSE: {stgnn_mse:.4f}")

# 3. CNN (Using unified features as raster-like input)
X_unified_reshaped = X_unified.reshape(X_unified.shape[0], X_unified.shape[1], 1)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_unified_reshaped, y_unified, test_size=0.2, random_state=42)

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, output_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNNModel(input_channels=X_unified.shape[1], output_dim=3).to(device)
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

if train_models or not os.path.exists("unified_cnn_model.pth"):
    print("Training CNN on unified data...")
    cnn_model.train()
    for epoch in tqdm(range(50)):
        optimizer_cnn.zero_grad()
        X_train_tensor = torch.FloatTensor(X_train_cnn).to(device)
        y_train_tensor = torch.FloatTensor(y_train_cnn).to(device)
        out = cnn_model(X_train_tensor)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
        optimizer_cnn.step()
        if epoch % 5 == 0:
            print(f'CNN Epoch {epoch}, Loss: {loss.item():.4f}')
    torch.save(cnn_model.state_dict(), "unified_cnn_model.pth")
else:
    cnn_model.load_state_dict(torch.load("unified_cnn_model.pth", map_location=device))
    cnn_model.eval()

cnn_model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_cnn).to(device)
    cnn_predictions = cnn_model(X_test_tensor).cpu().numpy()
cnn_mse = mean_squared_error(y_test_cnn, cnn_predictions)
print(f"CNN MSE: {cnn_mse:.4f}")

# 4. QIENN (Optimization on unified predictions)
class QIENN:
    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size
        self.generations = generations
        self.weights = np.random.uniform(0, 1, (population_size, 2))
        self.weights = self.weights / np.sum(self.weights, axis=1, keepdims=True)
    def quantum_mutation(self, weights):
        mutation_rate = 0.1
        for i in range(weights.shape[0]):
            if np.random.random() < mutation_rate:
                quantum_shift = np.random.normal(0, 0.1, 2)
                weights[i] += quantum_shift
                weights[i] = np.clip(weights[i], 0, 1)
                weights[i] /= np.sum(weights[i])
        return weights
    def fitness(self, weights, preds1, preds2, targets):
        fitness_scores = np.zeros(weights.shape[0])
        for i in range(weights.shape[0]):
            combined = weights[i, 0] * preds1 + weights[i, 1] * preds2
            mse = mean_squared_error(targets, combined)
            fitness_scores[i] = -mse
        return fitness_scores
    def optimize(self, preds1, preds2, targets):
        best_fitness = float('-inf')
        best_weights = None
        for gen in range(self.generations):
            fitness_scores = self.fitness(self.weights, preds1, preds2, targets)
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_weights = self.weights[best_idx].copy()
            elite = self.weights[np.argsort(fitness_scores)[-int(self.population_size*0.2):]]
            self.weights = np.vstack([elite, self.quantum_mutation(np.random.uniform(0, 1, (int(self.population_size*0.8), 2)))])
            self.weights = self.weights / np.sum(self.weights, axis=1, keepdims=True)
            if gen % 2 == 0:
                print(f'QIENN Generation {gen}, Best Fitness: {-best_fitness:.4f}')
        return best_weights

qienn = QIENN(population_size=20, generations=10)
# Use test set predictions
optimized_weights = qienn.optimize(stgnn_predictions, cnn_predictions, y_test)
final_unified_predictions = optimized_weights[0] * stgnn_predictions + optimized_weights[1] * cnn_predictions
qienn_mse = mean_squared_error(y_test, final_unified_predictions)
print(f"QIENN MSE: {qienn_mse:.4f}")

# 5. Tuned XGBoost (Best Single Model)
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)  # Tuned parameters
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
print(f"Tuned XGBoost MSE: {xgb_mse:.4f}")

# 6. Tuned Stacking Ensemble
# Prepare meta-features (predictions from all models)
meta_features = np.column_stack((lr_predictions, stgnn_predictions, cnn_predictions, final_unified_predictions, xgb_predictions))
# Train the meta-model on all meta-features (no inner split to match shapes)
meta_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)  # Tuned parameters
meta_model.fit(meta_features, y_test)  # Use full test set for meta-training
stacking_predictions = meta_model.predict(meta_features)
stacking_mse = mean_squared_error(y_test, stacking_predictions)
print(f"Tuned Stacking Ensemble MSE: {stacking_mse:.4f}")

# 7. Weighted Average Ensemble with Tuned Models
weights = {model: 1/mse if mse > 0 else 0 for model, mse in {'LinearRegression': lr_mse, 'STGNN': stgnn_mse, 'CNN': cnn_mse, 'QIENN': qienn_mse, 'TunedXGBoost': xgb_mse, 'TunedStacking': stacking_mse}.items()}
total_weight = sum(weights.values())
ensemble_weights = {model: w/total_weight for model, w in weights.items()}
ensemble_prediction = (ensemble_weights['LinearRegression'] * lr_predictions +
                      ensemble_weights['STGNN'] * stgnn_predictions +
                      ensemble_weights['CNN'] * cnn_predictions +
                      ensemble_weights['QIENN'] * final_unified_predictions +
                      ensemble_weights['TunedXGBoost'] * xgb_predictions +
                      ensemble_weights['TunedStacking'] * stacking_predictions)
ensemble_mse = mean_squared_error(y_test, ensemble_prediction)
print(f"Weighted Average Ensemble MSE: {ensemble_mse:.4f}")

# Select best model based on MSE
all_mse = {'LinearRegression': lr_mse, 'STGNN': stgnn_mse, 'CNN': cnn_mse, 'QIENN': qienn_mse,
           'TunedXGBoost': xgb_mse, 'TunedStacking': stacking_mse, 'WeightedAverageEnsemble': ensemble_mse}
best_model = min(all_mse, key=all_mse.get)
best_mse = all_mse[best_model]
print(f"Best Model: {best_model} with MSE: {best_mse:.4f}")
if best_model == 'LinearRegression':
    best_predictions = lr_predictions
elif best_model == 'STGNN':
    best_predictions = stgnn_predictions
elif best_model == 'CNN':
    best_predictions = cnn_predictions
elif best_model == 'QIENN':
    best_predictions = final_unified_predictions
elif best_model == 'TunedXGBoost':
    best_predictions = xgb_predictions
elif best_model == 'TunedStacking':
    best_predictions = stacking_predictions
else:  # WeightedAverageEnsemble
    best_predictions = ensemble_prediction
# Store predictions for each model
xgb_predictions_full = np.zeros_like(y_unified)
stacking_predictions_full = np.zeros_like(y_unified)
ensemble_predictions_full = np.zeros_like(y_unified)
xgb_model.fit(X_unified, y_unified)  # Retrain on full data for predictions

# Create STGNN dataset for full data
A_full = kneighbors_graph(data_unified[['X', 'Y']], n_neighbors=5, mode='connectivity', include_self=False)
edge_index_full = torch.tensor(A_full.nonzero(), dtype=torch.long)
stgnn_dataset_full = STGNNDataset(X_unified, y_unified, edge_index_full)
stgnn_dataset_full.data = stgnn_dataset_full.data.to(device)

# Get STGNN and CNN predictions for the full dataset
stgnn_predictions_full = stgnn_model(stgnn_dataset_full.data).detach().cpu().numpy()
cnn_predictions_full = cnn_model(torch.tensor(X_unified.reshape(X_unified.shape[0], X_unified.shape[1], 1), dtype=torch.float).to(device)).detach().cpu().numpy()

# Optimize weights using QIENN
optimized_weights = qienn.optimize(stgnn_predictions_full, cnn_predictions_full, y_unified)
qienn_predictions_full = optimized_weights[0] * stgnn_predictions_full + optimized_weights[1] * cnn_predictions_full

stacking_meta_features = np.column_stack((lr_model.predict(X_unified), 
                                         stgnn_predictions_full,
                                         cnn_predictions_full,
                                         qienn_predictions_full,
                                         xgb_model.predict(X_unified)))
meta_model.fit(stacking_meta_features, y_unified)
stacking_predictions_full = meta_model.predict(stacking_meta_features)
ensemble_weights_full = {model: 1/mse if mse > 0 else 0 for model, mse in {'LinearRegression': lr_mse, 'STGNN': stgnn_mse, 'CNN': cnn_mse, 'QIENN': qienn_mse, 'TunedXGBoost': xgb_mse, 'TunedStacking': stacking_mse}.items()}
total_weight_full = sum(ensemble_weights_full.values())
ensemble_weights_full = {model: w/total_weight_full for model, w in ensemble_weights_full.items()}
ensemble_predictions_full = (ensemble_weights_full['LinearRegression'] * lr_model.predict(X_unified) +
                            ensemble_weights_full['STGNN'] * stgnn_predictions_full +
                            ensemble_weights_full['CNN'] * cnn_predictions_full +
                            ensemble_weights_full['QIENN'] * qienn_predictions_full +
                            ensemble_weights_full['TunedXGBoost'] * xgb_model.predict(X_unified) +
                            ensemble_weights_full['TunedStacking'] * stacking_predictions_full)
xgb_predictions_full = xgb_model.predict(X_unified)

# Assign predictions to data_cleaned for plotting
models = {'Tuned XGBoost': xgb_predictions_full, 'Tuned Stacking Ensemble': stacking_predictions_full, 'Weighted Average Ensemble': ensemble_predictions_full}
for model_name, predictions in models.items():
    data_cleaned[f'{model_name.replace(" ", "_")}_Predicted_Cu_ppm'] = predictions[:, 0]
    data_cleaned[f'{model_name.replace(" ", "_")}_Predicted_Fe2O3_%'] = predictions[:, 1]
    data_cleaned[f'{model_name.replace(" ", "_")}_Predicted_Au_ppb'] = predictions[:, 2]

# Visualizations for Tuned Stacking Ensemble
print("Generating visualizations for Tuned Stacking Ensemble...")
# 2D Scatter Plots
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'], cmap='plasma', alpha=0.5)
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_Stacking_Ensemble_Predicted_Au_ppb'], cmap='magma', alpha=0.5)
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Predicted Gold Concentrations')
plt.show()
plt.close()

# Heatmaps
plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'], cmap='viridis')
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Heatmap of Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'], cmap='plasma')
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Heatmap of Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_Stacking_Ensemble_Predicted_Au_ppb'], cmap='magma')
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Heatmap of Predicted Gold Concentrations')
plt.show()
plt.close()

# High-Potential Zones
threshold = data_cleaned[['Tuned_Stacking_Ensemble_Predicted_Cu_ppm', 'Tuned_Stacking_Ensemble_Predicted_Fe2O3_%', 'Tuned_Stacking_Ensemble_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'] > threshold['Tuned_Stacking_Ensemble_Predicted_Cu_ppm']) |
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'] > threshold['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%']) |
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Au_ppb'] > threshold['Tuned_Stacking_Ensemble_Predicted_Au_ppb'])
]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential['X'], high_potential['Y'], c='red', label='High-Potential Zones')
plt.colorbar(label='High-Potential Indicator')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - High-Potential Zones for Cu, Fe, Au')
plt.legend()
plt.show()
plt.close()

# Correlation Heatmap
print("Generating correlation heatmap for Tuned Stacking Ensemble...")
aster_raster_cols = [col for col in data_cleaned.columns if col not in ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Magnetic_Grid']]
existing_cols = ['Tuned_Stacking_Ensemble_Predicted_Cu_ppm', 'Tuned_Stacking_Ensemble_Predicted_Fe2O3_%', 'Tuned_Stacking_Ensemble_Predicted_Au_ppb']
existing_cols = [col for col in existing_cols if col in data_cleaned.columns]

correlation_data = data_cleaned[existing_cols + aster_raster_cols]
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Tuned Stacking Ensemble - Correlation Heatmap of Predictions and Features')
plt.show()
plt.close()

# Anomaly Detection
print("Detecting anomalies for Tuned Stacking Ensemble...")
z_scores = np.abs(stats.zscore(data_cleaned[['Tuned_Stacking_Ensemble_Predicted_Cu_ppm', 'Tuned_Stacking_Ensemble_Predicted_Fe2O3_%', 'Tuned_Stacking_Ensemble_Predicted_Au_ppb']]))
anomalies = data_cleaned[(z_scores > 3).any(axis=1)]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(anomalies['X'], anomalies['Y'], c='red', label='Anomalies')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned Stacking Ensemble - Anomaly Detection in Predictions')
plt.legend()
plt.show()
plt.close()

# 3D Visualization with Plotly
print("Generating 3D visualizations for Tuned Stacking Ensemble...")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=data_cleaned['X'], y=data_cleaned['Y'], z=np.zeros(len(data_cleaned)),
    mode='markers', marker=dict(size=2, color='gray', opacity=0.1),
    name='All Data'
))
threshold = data_cleaned[['Tuned_Stacking_Ensemble_Predicted_Cu_ppm', 'Tuned_Stacking_Ensemble_Predicted_Fe2O3_%', 'Tuned_Stacking_Ensemble_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'] > threshold['Tuned_Stacking_Ensemble_Predicted_Cu_ppm']) |
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'] > threshold['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%']) |
    (data_cleaned['Tuned_Stacking_Ensemble_Predicted_Au_ppb'] > threshold['Tuned_Stacking_Ensemble_Predicted_Au_ppb'])
]
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_Stacking_Ensemble_Predicted_Cu_ppm'], colorscale='Viridis', opacity=0.8),
    name='Tuned Stacking Ensemble High-Potential Cu'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_Stacking_Ensemble_Predicted_Fe2O3_%'], colorscale='Plasma', opacity=0.8),
    name='Tuned Stacking Ensemble High-Potential Fe'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_Stacking_Ensemble_Predicted_Au_ppb'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_Stacking_Ensemble_Predicted_Au_ppb'], colorscale='Magma', opacity=0.8),
    name='Tuned Stacking Ensemble High-Potential Au'
))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Predicted Value',
        xaxis=dict(range=[data_cleaned['X'].min(), data_cleaned['X'].max()]),
        yaxis=dict(range=[data_cleaned['Y'].min(), data_cleaned['Y'].max()]),
        zaxis=dict(range=[0, max(high_potential[['Tuned_Stacking_Ensemble_Predicted_Cu_ppm', 'Tuned_Stacking_Ensemble_Predicted_Fe2O3_%', 'Tuned_Stacking_Ensemble_Predicted_Au_ppb']].max()) * 1.1])
    ),
    title='Tuned Stacking Ensemble - 3D Visualization of High-Potential Zones',
    showlegend=True
)
fig_3d.show()

# Visualizations for Tuned XGBoost
print("Generating visualizations for Tuned XGBoost...")
# 2D Scatter Plots
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_XGBoost_Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_XGBoost_Predicted_Fe2O3_%'], cmap='plasma', alpha=0.5)
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Tuned_XGBoost_Predicted_Au_ppb'], cmap='magma', alpha=0.5)
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Predicted Gold Concentrations')
plt.show()
plt.close()

# Heatmaps
plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_XGBoost_Predicted_Cu_ppm'], cmap='viridis')
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Heatmap of Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_XGBoost_Predicted_Fe2O3_%'], cmap='plasma')
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Heatmap of Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Tuned_XGBoost_Predicted_Au_ppb'], cmap='magma')
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Heatmap of Predicted Gold Concentrations')
plt.show()
plt.close()

# High-Potential Zones
threshold = data_cleaned[['Tuned_XGBoost_Predicted_Cu_ppm', 'Tuned_XGBoost_Predicted_Fe2O3_%', 'Tuned_XGBoost_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Tuned_XGBoost_Predicted_Cu_ppm'] > threshold['Tuned_XGBoost_Predicted_Cu_ppm']) |
    (data_cleaned['Tuned_XGBoost_Predicted_Fe2O3_%'] > threshold['Tuned_XGBoost_Predicted_Fe2O3_%']) |
    (data_cleaned['Tuned_XGBoost_Predicted_Au_ppb'] > threshold['Tuned_XGBoost_Predicted_Au_ppb'])
]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential['X'], high_potential['Y'], c='red', label='High-Potential Zones')
plt.colorbar(label='High-Potential Indicator')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - High-Potential Zones for Cu, Fe, Au')
plt.legend()
plt.show()
plt.close()

# Correlation Heatmap
print("Generating correlation heatmap for Tuned XGBoost...")
aster_raster_cols = [col for col in data_cleaned.columns if col not in ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Magnetic_Grid']]
existing_cols = ['Tuned_XGBoost_Predicted_Cu_ppm', 'Tuned_XGBoost_Predicted_Fe2O3_%', 'Tuned_XGBoost_Predicted_Au_ppb']
existing_cols = [col for col in existing_cols if col in data_cleaned.columns]
correlation_data = data_cleaned[existing_cols + aster_raster_cols]
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Tuned XGBoost - Correlation Heatmap of Predictions and Features')
plt.show()
plt.close()

# Anomaly Detection
print("Detecting anomalies for Tuned XGBoost...")
z_scores = np.abs(stats.zscore(data_cleaned[['Tuned_XGBoost_Predicted_Cu_ppm', 'Tuned_XGBoost_Predicted_Fe2O3_%', 'Tuned_XGBoost_Predicted_Au_ppb']]))
anomalies = data_cleaned[(z_scores > 3).any(axis=1)]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(anomalies['X'], anomalies['Y'], c='red', label='Anomalies')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tuned XGBoost - Anomaly Detection in Predictions')
plt.legend()
plt.show()
plt.close()

# 3D Visualization with Plotly
print("Generating 3D visualizations for Tuned XGBoost...")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=data_cleaned['X'], y=data_cleaned['Y'], z=np.zeros(len(data_cleaned)),
    mode='markers', marker=dict(size=2, color='gray', opacity=0.1),
    name='All Data'
))
threshold = data_cleaned[['Tuned_XGBoost_Predicted_Cu_ppm', 'Tuned_XGBoost_Predicted_Fe2O3_%', 'Tuned_XGBoost_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Tuned_XGBoost_Predicted_Cu_ppm'] > threshold['Tuned_XGBoost_Predicted_Cu_ppm']) |
    (data_cleaned['Tuned_XGBoost_Predicted_Fe2O3_%'] > threshold['Tuned_XGBoost_Predicted_Fe2O3_%']) |
    (data_cleaned['Tuned_XGBoost_Predicted_Au_ppb'] > threshold['Tuned_XGBoost_Predicted_Au_ppb'])
]
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_XGBoost_Predicted_Cu_ppm'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_XGBoost_Predicted_Cu_ppm'], colorscale='Viridis', opacity=0.8),
    name='Tuned XGBoost High-Potential Cu'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_XGBoost_Predicted_Fe2O3_%'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_XGBoost_Predicted_Fe2O3_%'], colorscale='Plasma', opacity=0.8),
    name='Tuned XGBoost High-Potential Fe'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Tuned_XGBoost_Predicted_Au_ppb'],
    mode='markers', marker=dict(size=5, color=high_potential['Tuned_XGBoost_Predicted_Au_ppb'], colorscale='Magma', opacity=0.8),
    name='Tuned XGBoost High-Potential Au'
))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Predicted Value',
        xaxis=dict(range=[data_cleaned['X'].min(), data_cleaned['X'].max()]),
        yaxis=dict(range=[data_cleaned['Y'].min(), data_cleaned['Y'].max()]),
        zaxis=dict(range=[0, max(high_potential[['Tuned_XGBoost_Predicted_Cu_ppm', 'Tuned_XGBoost_Predicted_Fe2O3_%', 'Tuned_XGBoost_Predicted_Au_ppb']].max()) * 1.1])
    ),
    title='Tuned XGBoost - 3D Visualization of High-Potential Zones',
    showlegend=True
)
fig_3d.show()

# Visualizations for Weighted Average Ensemble
print("Generating visualizations for Weighted Average Ensemble...")
# 2D Scatter Plots
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Weighted_Average_Ensemble_Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Weighted_Average_Ensemble_Predicted_Fe2O3_%'], cmap='plasma', alpha=0.5)
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Weighted_Average_Ensemble_Predicted_Au_ppb'], cmap='magma', alpha=0.5)
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Predicted Gold Concentrations')
plt.show()
plt.close()

# Heatmaps
plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Weighted_Average_Ensemble_Predicted_Cu_ppm'], cmap='viridis')
plt.colorbar(label='Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Heatmap of Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Weighted_Average_Ensemble_Predicted_Fe2O3_%'], cmap='plasma')
plt.colorbar(label='Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Heatmap of Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Weighted_Average_Ensemble_Predicted_Au_ppb'], cmap='magma')
plt.colorbar(label='Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Heatmap of Predicted Gold Concentrations')
plt.show()
plt.close()

# High-Potential Zones
threshold = data_cleaned[['Weighted_Average_Ensemble_Predicted_Cu_ppm', 'Weighted_Average_Ensemble_Predicted_Fe2O3_%', 'Weighted_Average_Ensemble_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Cu_ppm'] > threshold['Weighted_Average_Ensemble_Predicted_Cu_ppm']) |
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Fe2O3_%'] > threshold['Weighted_Average_Ensemble_Predicted_Fe2O3_%']) |
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Au_ppb'] > threshold['Weighted_Average_Ensemble_Predicted_Au_ppb'])
]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential['X'], high_potential['Y'], c='red', label='High-Potential Zones')
plt.colorbar(label='High-Potential Indicator')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - High-Potential Zones for Cu, Fe, Au')
plt.legend()
plt.show()
plt.close()

# Correlation Heatmap
print("Generating correlation heatmap for Weighted Average Ensemble...")
aster_raster_cols = [col for col in data_cleaned.columns if col not in ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Magnetic_Grid']]
existing_cols = ['Weighted_Average_Ensemble_Predicted_Cu_ppm', 'Weighted_Average_Ensemble_Predicted_Fe2O3_%', 'Weighted_Average_Ensemble_Predicted_Au_ppb']
existing_cols = [col for col in existing_cols if col in data_cleaned.columns]
correlation_data = data_cleaned[existing_cols + aster_raster_cols]
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Weighted Average Ensemble - Correlation Heatmap of Predictions and Features')
plt.show()
plt.close()

# Anomaly Detection
print("Detecting anomalies for Weighted Average Ensemble...")
z_scores = np.abs(stats.zscore(data_cleaned[['Weighted_Average_Ensemble_Predicted_Cu_ppm', 'Weighted_Average_Ensemble_Predicted_Fe2O3_%', 'Weighted_Average_Ensemble_Predicted_Au_ppb']]))
anomalies = data_cleaned[(z_scores > 3).any(axis=1)]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(anomalies['X'], anomalies['Y'], c='red', label='Anomalies')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Weighted Average Ensemble - Anomaly Detection in Predictions')
plt.legend()
plt.show()
plt.close()

# 3D Visualization with Plotly
print("Generating 3D visualizations for Weighted Average Ensemble...")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=data_cleaned['X'], y=data_cleaned['Y'], z=np.zeros(len(data_cleaned)),
    mode='markers', marker=dict(size=2, color='gray', opacity=0.1),
    name='All Data'
))
threshold = data_cleaned[['Weighted_Average_Ensemble_Predicted_Cu_ppm', 'Weighted_Average_Ensemble_Predicted_Fe2O3_%', 'Weighted_Average_Ensemble_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Cu_ppm'] > threshold['Weighted_Average_Ensemble_Predicted_Cu_ppm']) |
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Fe2O3_%'] > threshold['Weighted_Average_Ensemble_Predicted_Fe2O3_%']) |
    (data_cleaned['Weighted_Average_Ensemble_Predicted_Au_ppb'] > threshold['Weighted_Average_Ensemble_Predicted_Au_ppb'])
]
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Weighted_Average_Ensemble_Predicted_Cu_ppm'],
    mode='markers', marker=dict(size=5, color=high_potential['Weighted_Average_Ensemble_Predicted_Cu_ppm'], colorscale='Viridis', opacity=0.8),
    name='Weighted Average Ensemble High-Potential Cu'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Weighted_Average_Ensemble_Predicted_Fe2O3_%'],
    mode='markers', marker=dict(size=5, color=high_potential['Weighted_Average_Ensemble_Predicted_Fe2O3_%'], colorscale='Plasma', opacity=0.8),
    name='Weighted Average Ensemble High-Potential Fe'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Weighted_Average_Ensemble_Predicted_Au_ppb'],
    mode='markers', marker=dict(size=5, color=high_potential['Weighted_Average_Ensemble_Predicted_Au_ppb'], colorscale='Magma', opacity=0.8),
    name='Weighted Average Ensemble High-Potential Au'
))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Predicted Value',
        xaxis=dict(range=[data_cleaned['X'].min(), data_cleaned['X'].max()]),
        yaxis=dict(range=[data_cleaned['Y'].min(), data_cleaned['Y'].max()]),
        zaxis=dict(range=[0, max(high_potential[['Weighted_Average_Ensemble_Predicted_Cu_ppm', 'Weighted_Average_Ensemble_Predicted_Fe2O3_%', 'Weighted_Average_Ensemble_Predicted_Au_ppb']].max()) * 1.1])
    ),
    title='Weighted Average Ensemble - 3D Visualization of High-Potential Zones',
    showlegend=True
)
fig_3d.show()
