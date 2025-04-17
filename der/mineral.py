# Import libraries
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

# Define base directory (updated to local path in VS Code environment)
base_dir = "Processed_Mineral_Data/"  # Relative to the script's location in der/

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

# --- Create Graph Data for STGNN ---
class STGNNDataset:
    def __init__(self, data):
        # Check for NaN or inf
        if data.isna().any().any() or np.isinf(data.values).any():
            raise ValueError("Data contains NaN or inf values")
        
        self.x = torch.tensor(data.drop(columns=['Cu_ppm', 'Fe2O3_%', 'Au_ppb']).values, dtype=torch.float)
        self.y = torch.tensor(data[['Cu_ppm', 'Fe2O3_%', 'Au_ppb']].values, dtype=torch.float)
        
        # Create adjacency matrix
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(data[['X', 'Y']], n_neighbors=3, mode='connectivity', include_self=False)
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        
        if edge_index.shape[1] == 0:
            raise ValueError("Empty edge_index, check graph construction")
        
        self.edge_index = edge_index
        self.data = Data(x=self.x, edge_index=self.edge_index, y=self.y)

try:
    dataset = STGNNDataset(data_cleaned_scaled)
    print("Graph data created. Edge count:", dataset.edge_index.shape[1])
except Exception as e:
    print(f"Error in graph construction: {e}")
    raise

# --- Define STGNN Model ---
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

# --- Train or Load STGNN ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STGNN(input_dim=dataset.x.shape[1], hidden_dim=16, output_dim=3).to(device)
dataset.data = dataset.data.to(device)
optimizer_stgnn = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

if train_models or not os.path.exists(model_save_path_stgnn):
    print("Training STGNN...")
    def train_stgnn():
        model.train()
        optimizer_stgnn.zero_grad()
        out = model(dataset.data)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN or inf in STGNN output, skipping step")
            return float('inf')
        loss = criterion(out, dataset.data.y)
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or inf loss in STGNN, skipping step")
            return float('inf')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_stgnn.step()
        return loss.item()

    for epoch in tqdm(range(50)):
        loss = train_stgnn()
        if loss == float('inf'):
            continue
        if epoch % 5 == 0:
            print(f'STGNN Epoch {epoch}, Loss: {loss:.4f}')
    torch.save(model.state_dict(), model_save_path_stgnn)
    print(f"STGNN model saved to {model_save_path_stgnn}")
else:
    model.load_state_dict(torch.load(model_save_path_stgnn, map_location=device))
    model.eval()
    print(f"STGNN model loaded from {model_save_path_stgnn}")

# --- Create Raster Stack for CNN ---
print("Creating raster stack...")
raster_stack = []
for key in aster_rasters.keys():
    values = data_cleaned[key].values
    values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    raster_stack.append(values.reshape(-1, 1))
raster_stack.append(np.nan_to_num(data_cleaned['Magnetic_Grid'].values, nan=0, posinf=0, neginf=0).reshape(-1, 1))
raster_stack = np.hstack(raster_stack)  # Shape: (samples, channels)

expected_channels = 324
if raster_stack.shape[1] < expected_channels:
    padding = np.zeros((raster_stack.shape[0], expected_channels - raster_stack.shape[1]))
    raster_stack = np.hstack((raster_stack, padding))
elif raster_stack.shape[1] > expected_channels:
    raster_stack = raster_stack[:, :expected_channels]
# Normalize raster stack
raster_stack = (raster_stack - np.nanmean(raster_stack, axis=0, keepdims=True)) / (np.nanstd(raster_stack, axis=0, keepdims=True) + 1e-8)

# Reshape for CNN: (samples, channels, sequence_length=1)
raster_stack = raster_stack.reshape(raster_stack.shape[0], raster_stack.shape[1], 1)

# Prepare targets for CNN
cnn_targets = target.values
# Align cnn_targets with raster_stack
cnn_targets = cnn_targets[data_cleaned.index]

# Convert to PyTorch tensors
raster_tensor = torch.FloatTensor(raster_stack)
target_tensor = torch.FloatTensor(cnn_targets)

# Create DataLoader
dataset_cnn = TensorDataset(raster_tensor, target_tensor)
dataloader = DataLoader(dataset_cnn, batch_size=32, shuffle=True)

# --- Define CNN Model ---
class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=1, padding=0)  # Sequence length=1
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

# --- Train or Load CNN ---
cnn_model = CNNModel(input_channels=raster_stack.shape[1], output_dim=3).to(device)
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=0.001)  # Reduced learning rate
criterion = nn.MSELoss()

if train_models or not os.path.exists(model_save_path_cnn):
    print("Training CNN...")
    cnn_model.train()
    for epoch in tqdm(range(50)):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_cnn.zero_grad()
            output = cnn_model(batch_x)  # Shape: (batch_size, channels, 1)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN or inf in CNN output, skipping step")
                continue
            loss = criterion(output, batch_y)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or inf loss in CNN, skipping step")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            optimizer_cnn.step()
            total_loss += loss.item()
        if epoch % 5 == 0 and total_loss > 0:
            print(f'CNN Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
    torch.save(cnn_model.state_dict(), model_save_path_cnn)
    print(f"CNN model saved to {model_save_path_cnn}")
else:
    cnn_model.load_state_dict(torch.load(model_save_path_cnn, map_location=device))
    cnn_model.eval()
    print(f"CNN model loaded from {model_save_path_cnn}")

# --- Combine STGNN and CNN Predictions ---
print("Combining STGNN and CNN predictions...")
model.eval()
with torch.no_grad():
    stgnn_predictions = model(dataset.data).cpu().numpy()
cnn_model.eval()
with torch.no_grad():
    cnn_predictions = cnn_model(raster_tensor.to(device)).cpu().numpy()
combined_predictions = (stgnn_predictions + cnn_predictions) / 2
data_cleaned[['Combined_Predicted_Cu_ppm', 'Combined_Predicted_Fe2O3_%', 'Combined_Predicted_Au_ppb']] = combined_predictions

# --- QIENN Optimization ---
print("Optimizing with QIENN...")
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

    def fitness(self, weights, stgnn_preds, cnn_preds, targets):
        fitness_scores = np.zeros(weights.shape[0])
        for i in range(weights.shape[0]):
            combined = weights[i, 0] * stgnn_preds + weights[i, 1] * cnn_preds
            mse = np.mean((combined - targets) ** 2)
            fitness_scores[i] = -mse  # Negative MSE for maximization
        return fitness_scores

    def optimize(self, stgnn_preds, cnn_preds, targets):
        best_fitness = float('-inf')
        best_weights = None
        for gen in range(self.generations):
            fitness_scores = self.fitness(self.weights, stgnn_preds, cnn_preds, targets)
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

# Initialize and optimize
qienn = QIENN(population_size=20, generations=10)
optimized_weights = qienn.optimize(stgnn_predictions, cnn_predictions, target.values)
final_predictions = optimized_weights[0] * stgnn_predictions + optimized_weights[1] * cnn_predictions
data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']] = final_predictions

print("Evaluation model performance:...")
from sklearn.metrics import mean_squared_error
true_targets = target.values

final_mse = mean_squared_error(true_targets, data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']].values)
print(f"Final Prediction MSE: {final_mse:.4f}")
threshold_true = data_cleaned[['Cu_ppm', 'Fe2O3_%', 'Au_ppb']].quantile(0.95)
threshold_pred = data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']].quantile(0.95)
high_potential_true = (
    (data_cleaned['Cu_ppm'] > threshold_true['Cu_ppm']) |
    (data_cleaned['Fe2O3_%'] > threshold_true['Fe2O3_%']) |
    (data_cleaned['Au_ppb'] > threshold_true['Au_ppb'])
)
high_potential_pred = (
    (data_cleaned['Final_Predicted_Cu_ppm'] > threshold_pred['Final_Predicted_Cu_ppm']) |
    (data_cleaned['Final_Predicted_Fe2O3_%'] > threshold_pred['Final_Predicted_Fe2O3_%']) |
    (data_cleaned['Final_Predicted_Au_ppb'] > threshold_pred['Final_Predicted_Au_ppb'])
)
accuracy = (high_potential_true == high_potential_pred).mean()
print(f"Accuracy (High-Potential Classification): {accuracy:.4f}")
# --- Visualize Combined Predictions ---
# Copper (Cu) Plot
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Final_Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Final Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Final Predicted Copper Concentrations')
plt.show()
plt.close()

# Iron (Fe) Plot
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Final_Predicted_Fe2O3_%'], cmap='plasma', alpha=0.5)
plt.colorbar(label='Final Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Final Predicted Iron Concentrations')
plt.show()
plt.close()

# Gold (Au) Plot
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Final_Predicted_Au_ppb'], cmap='magma', alpha=0.5)
plt.colorbar(label='Final Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Final Predicted Gold Concentrations')
plt.show()
plt.close()

# Heatmaps
plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Final_Predicted_Cu_ppm'], cmap='viridis')
plt.colorbar(label='Final Predicted Copper (Cu_ppm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Final Predicted Copper Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Final_Predicted_Fe2O3_%'], cmap='plasma')
plt.colorbar(label='Final Predicted Iron (Fe2O3_%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Final Predicted Iron Concentrations')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.tricontourf(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Final_Predicted_Au_ppb'], cmap='magma')
plt.colorbar(label='Final Predicted Gold (Au_ppb)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Final Predicted Gold Concentrations')
plt.show()
plt.close()

# High-Potential Zones (e.g., top 5% of predictions)
threshold = data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Final_Predicted_Cu_ppm'] > threshold['Final_Predicted_Cu_ppm']) |
    (data_cleaned['Final_Predicted_Fe2O3_%'] > threshold['Final_Predicted_Fe2O3_%']) |
    (data_cleaned['Final_Predicted_Au_ppb'] > threshold['Final_Predicted_Au_ppb'])
]

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(high_potential['X'], high_potential['Y'], c='red', label='High-Potential Zones')
plt.colorbar(label='High-Potential Indicator')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('High-Potential Zones for Cu, Fe, Au')
plt.legend()
plt.show()
plt.close()

# --- Extra Features ---

# 1. Correlation Heatmap
# 1. Correlation Heatmap
print("Generating correlation heatmap...")
# Handle case where aster_rasters might be empty when loading from pickle
aster_raster_cols = [col for col in data_cleaned.columns if col not in relevant_columns + ['Magnetic_Grid', 'Magnetic_Anomaly', 'Combined_Predicted_Cu_ppm', 'Combined_Predicted_Fe2O3_%', 'Combined_Predicted_Au_ppb', 'Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']]
# Select only existing columns
base_cols = ['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']
existing_cols = [col for col in base_cols + ['Magnetic_Anomaly'] if col in data_cleaned.columns]
correlation_data = data_cleaned[existing_cols + aster_raster_cols]
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Predictions and Features')
plt.show()
plt.close()

# 2. Anomaly Detection
print("Detecting anomalies...")
z_scores = np.abs(stats.zscore(data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']]))
anomalies = data_cleaned[z_scores > 3]
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
plt.scatter(anomalies['X'], anomalies['Y'], c='red', label='Anomalies')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Anomaly Detection in Predictions')
plt.legend()
plt.show()
plt.close()

# --- 3D Visualization with Plotly ---
print("Generating 3D visualizations...")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=data_cleaned['X'], y=data_cleaned['Y'], z=np.zeros(len(data_cleaned)),
    mode='markers', marker=dict(size=2, color='gray', opacity=0.1),
    name='All Data'
))
threshold = data_cleaned[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']].quantile(0.95)
high_potential = data_cleaned[
    (data_cleaned['Final_Predicted_Cu_ppm'] > threshold['Final_Predicted_Cu_ppm']) |
    (data_cleaned['Final_Predicted_Fe2O3_%'] > threshold['Final_Predicted_Fe2O3_%']) |
    (data_cleaned['Final_Predicted_Au_ppb'] > threshold['Final_Predicted_Au_ppb'])
]
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Final_Predicted_Cu_ppm'],
    mode='markers', marker=dict(size=5, color=high_potential['Final_Predicted_Cu_ppm'], colorscale='Viridis', opacity=0.8),
    name='High-Potential Cu'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Final_Predicted_Fe2O3_%'],
    mode='markers', marker=dict(size=5, color=high_potential['Final_Predicted_Fe2O3_%'], colorscale='Plasma', opacity=0.8),
    name='High-Potential Fe'
))
fig_3d.add_trace(go.Scatter3d(
    x=high_potential['X'], y=high_potential['Y'], z=high_potential['Final_Predicted_Au_ppb'],
    mode='markers', marker=dict(size=5, color=high_potential['Final_Predicted_Au_ppb'], colorscale='Magma', opacity=0.8),
    name='High-Potential Au'
))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Predicted Value',
        xaxis=dict(range=[data_cleaned['X'].min(), data_cleaned['X'].max()]),
        yaxis=dict(range=[data_cleaned['Y'].min(), data_cleaned['Y'].max()]),
        zaxis=dict(range=[0, max(high_potential[['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']].max()) * 1.1])
    ),
    title='3D Visualization of High-Potential Zones',
    showlegend=True
)
fig_3d.show()

# --- Interactive Dashboard ---
# --- Interactive Dashboard ---
print("Setting up interactive dashboard...")
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Mineral Prediction Dashboard"),
    dcc.Dropdown(
        id='mineral-dropdown',
        options=[{'label': m, 'value': m} for m in ['Final_Predicted_Cu_ppm', 'Final_Predicted_Fe2O3_%', 'Final_Predicted_Au_ppb']],
        value='Final_Predicted_Cu_ppm'
    ),
    dcc.Graph(id='prediction-graph')
])

@app.callback(
    Output('prediction-graph', 'figure'),
    Input('mineral-dropdown', 'value')
)
def update_graph(selected_mineral):
    fig = go.Figure(data=go.Scatter(
        x=data_cleaned['X'], y=data_cleaned['Y'], mode='markers',
        marker=dict(size=5, color=data_cleaned[selected_mineral], colorscale='Viridis', opacity=0.6)
    ))
    fig.update_layout(title=f'Interactive Plot of {selected_mineral}', xaxis_title='X Coordinate', yaxis_title='Y Coordinate')
    return fig

if __name__ == '__main__':
    print("Dash server starting on http://127.0.0.1:8050...")
    import webbrowser
    webbrowser.open('http://127.0.0.1:8050/')
    app.run(debug=True, port=8050)  # Updated to new syntax

# --- Final Report Preparation ---
print("Generating report...")
stgnn_loss = "N/A (Loaded)" if not train_models and os.path.exists(model_save_path_stgnn) else loss
cnn_loss = "N/A (Loaded)" if not train_models and os.path.exists(model_save_path_cnn) else total_loss/len(dataloader) if 'total_loss' in locals() else "N/A"
report = {
    "Data Summary": f"Total rows: {data_cleaned.shape[0]}, Columns: {data_cleaned.shape[1]}",
    "High-Potential Zones": high_potential.shape[0],
    "Anomalies Detected": anomalies.shape[0],
    "Final MSE": final_mse,
    "Accuracy (High-Potential Classification)": accuracy,
    "STGNN Final Loss": stgnn_loss,
    "CNN Final Loss": cnn_loss,
    "QIENN Optimized Weights": optimized_weights.tolist()
}
print(report)
import json
with open('mineral_report.json', 'w') as f:
    json.dump(report, f)
print("Report saved as mineral_report.json")