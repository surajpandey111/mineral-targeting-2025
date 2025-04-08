from flask import Flask, render_template, request, send_file, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load preprocessed data (run mineral_targeting.py first to generate)
file_path = "NGCM-Stream-Sediment-Analysis-Updated.xlsx"
data = pd.read_excel(file_path)
relevant_columns = ['X', 'Y', 'Cu_ppm', 'Fe2O3_%', 'Au_ppb', 'Ni_ppm', 'Zn_ppm', 'Pb_ppm']
data_selected = data[relevant_columns].dropna()

# Standardize features (as in original code)
scaler = StandardScaler()
features = data_selected[['X', 'Y', 'Ni_ppm', 'Zn_ppm', 'Pb_ppm']]
scaled_features = scaler.fit_transform(features)
data_selected[['X_scaled', 'Y_scaled', 'Ni_ppm_scaled', 'Zn_ppm_scaled', 'Pb_ppm_scaled']] = scaled_features

# Train models (reuse trained parameters from original)
X = data_selected[['X_scaled', 'Y_scaled', 'Ni_ppm_scaled', 'Zn_ppm_scaled', 'Pb_ppm_scaled']]
y_cu = data_selected['Cu_ppm']
y_fe = data_selected['Fe2O3_%']
y_au = data_selected['Au_ppb']
X_train_cu, X_test_cu, y_train_cu, y_test_cu = train_test_split(X, y_cu, test_size=0.2, random_state=42)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X, y_fe, test_size=0.2, random_state=42)
X_train_au, X_test_au, y_train_au, y_test_au = train_test_split(X, y_au, test_size=0.2, random_state=42)

model_cu = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model_fe = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model_au = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

model_cu.fit(X_train_cu, y_train_cu)
model_fe.fit(X_train_fe, y_train_fe)
model_au.fit(X_train_au, y_train_au)

# Precompute predictions (as in original)
data_selected['Predicted_Cu_ppm'] = model_cu.predict(X)
data_selected['Predicted_Fe2O3_%'] = model_fe.predict(X)
data_selected['Predicted_Au_ppb'] = model_au.predict(X)

# High-potential zones (as in original)
cu_threshold = data_selected['Predicted_Cu_ppm'].quantile(0.95)
fe_threshold = data_selected['Predicted_Fe2O3_%'].quantile(0.95)
au_threshold = data_selected['Predicted_Au_ppb'].quantile(0.95)
high_potential_cu = data_selected[data_selected['Predicted_Cu_ppm'] >= cu_threshold]
high_potential_fe = data_selected[data_selected['Predicted_Fe2O3_%'] >= fe_threshold]
high_potential_au = data_selected[data_selected['Predicted_Au_ppb'] >= au_threshold]

# Save CSV (as in original)
data_selected.to_csv('mineral_targeting_results_enhanced.csv', index=False)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    mse_cu = mean_squared_error(y_test_cu, model_cu.predict(X_test_cu))
    mse_fe = mean_squared_error(y_test_fe, model_fe.predict(X_test_fe))
    mse_au = mean_squared_error(y_test_au, model_au.predict(X_test_au))

    # Generate plots (reusing Matplotlib style)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Copper (Cu_ppm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Copper Concentrations')
    buf_cu = BytesIO()
    plt.savefig(buf_cu, format='png')
    buf_cu.seek(0)
    cu_plot = base64.b64encode(buf_cu.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Fe2O3_%'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Iron Oxide (Fe2O3_%)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Iron Oxide Concentrations')
    buf_fe = BytesIO()
    plt.savefig(buf_fe, format='png')
    buf_fe.seek(0)
    fe_plot = base64.b64encode(buf_fe.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Au_ppb'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Gold (Au_ppb)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Gold Concentrations')
    buf_au = BytesIO()
    plt.savefig(buf_au, format='png')
    buf_au.seek(0)
    au_plot = base64.b64encode(buf_au.read()).decode('utf-8')
    plt.close()

    return render_template('predict.html', mse_cu=mse_cu, mse_fe=mse_fe, mse_au=mse_au,
                          cu_plot=cu_plot, fe_plot=fe_plot, au_plot=au_plot,
                          cu_count=len(high_potential_cu), fe_count=len(high_potential_fe), au_count=len(high_potential_au))

@app.route('/download_csv')
def download_csv():
    return send_file('mineral_targeting_results_enhanced.csv', as_attachment=True)

@app.route('/save_plot/<mineral>')
def save_plot(mineral):
    if mineral == 'cu':
        plt.figure(figsize=(10, 6))
        plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Predicted Copper (Cu_ppm)')
        plt.title('Predicted Copper Concentrations')
        plt.savefig('static/cu_plot.png')
    elif mineral == 'fe':
        plt.figure(figsize=(10, 6))
        plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Fe2O3_%'], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
        plt.title('Predicted Iron Oxide Concentrations')
        plt.savefig('static/fe_plot.png')
    elif mineral == 'au':
        plt.figure(figsize=(10, 6))
        plt.scatter(data_selected['X'], data_selected['Y'], c=data_selected['Predicted_Au_ppb'], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Predicted Gold (Au_ppb)')
        plt.title('Predicted Gold Concentrations')
        plt.savefig('static/au_plot.png')
    plt.close()
    return f"Plot for {mineral} saved as static/{mineral}_plot.png"

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000)