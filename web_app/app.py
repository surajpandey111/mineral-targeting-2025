from flask import Flask, render_template, send_file, abort
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from joblib import load
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load precomputed data and models (use existing files instead of regenerating)

data_cleaned = pd.read_csv('mineral_targeting_results_enhanced.csv')  # Load precomputed CSV

# Load pre-trained models
model_cu = load('model_cu.joblib')
model_fe = load('model_fe.joblib')
model_au = load('model_au.joblib')

# No need to re-standardize or predict here since data is precomputed

# High-potential zones (recompute based on precomputed predictions)
cu_threshold = data_cleaned['Predicted_Cu_ppm'].quantile(0.95)
fe_threshold = data_cleaned['Predicted_Fe2O3_%'].quantile(0.95)
au_threshold = data_cleaned['Predicted_Au_ppb'].quantile(0.95)
high_potential_cu = data_cleaned[data_cleaned['Predicted_Cu_ppm'] >= cu_threshold]
high_potential_fe = data_cleaned[data_cleaned['Predicted_Fe2O3_%'] >= fe_threshold]
high_potential_au = data_cleaned[data_cleaned['Predicted_Au_ppb'] >= au_threshold]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Scatter plots (original distributions)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Cu_ppm'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Copper (Cu_ppm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Distribution of Copper Concentrations')
    buf_orig_cu = BytesIO()
    plt.savefig(buf_orig_cu, format='png')
    buf_orig_cu.seek(0)
    orig_cu_plot = base64.b64encode(buf_orig_cu.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Fe2O3_%'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Iron Oxide (Fe2O3_%)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Distribution of Iron Oxide Concentrations')
    buf_orig_fe = BytesIO()
    plt.savefig(buf_orig_fe, format='png')
    buf_orig_fe.seek(0)
    orig_fe_plot = base64.b64encode(buf_orig_fe.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Au_ppb'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Gold (Au_ppb)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Distribution of Gold Concentrations')
    buf_orig_au = BytesIO()
    plt.savefig(buf_orig_au, format='png')
    buf_orig_au.seek(0)
    orig_au_plot = base64.b64encode(buf_orig_au.read()).decode('utf-8')
    plt.close()

    # Scatter plots (predicted concentrations)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Copper (Cu_ppm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Copper Concentrations')
    buf_pred_cu = BytesIO()
    plt.savefig(buf_pred_cu, format='png')
    buf_pred_cu.seek(0)
    pred_cu_plot = base64.b64encode(buf_pred_cu.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Fe2O3_%'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Iron Oxide (Fe2O3_%)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Iron Oxide Concentrations')
    buf_pred_fe = BytesIO()
    plt.savefig(buf_pred_fe, format='png')
    buf_pred_fe.seek(0)
    pred_fe_plot = base64.b64encode(buf_pred_fe.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Au_ppb'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Predicted Gold (Au_ppb)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Gold Concentrations')
    buf_pred_au = BytesIO()
    plt.savefig(buf_pred_au, format='png')
    buf_pred_au.seek(0)
    pred_au_plot = base64.b64encode(buf_pred_au.read()).decode('utf-8')
    plt.close()

    # Heatmaps (predicted concentrations)
    def create_heatmap(x, y, values, title, label):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=values)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label=label)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return plot

    heatmap_cu_plot = create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Cu_ppm'], 'Heatmap of Predicted Copper Concentrations', 'Predicted Copper (Cu_ppm)')
    heatmap_fe_plot = create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Fe2O3_%'], 'Heatmap of Predicted Iron Oxide Concentrations', 'Predicted Iron Oxide (Fe2O3_%)')
    heatmap_au_plot = create_heatmap(data_cleaned['X'], data_cleaned['Y'], data_cleaned['Predicted_Au_ppb'], 'Heatmap of Predicted Gold Concentrations', 'Predicted Gold (Au_ppb)')

    # High-potential zones (scatter plots)
    plt.figure(figsize=(10, 6))
    plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
    plt.scatter(high_potential_cu['X'], high_potential_cu['Y'], c='red', label=f'Cu > {cu_threshold:.2f} ppm')
    plt.colorbar(label='Predicted Copper (Cu_ppm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('High-Potential Copper Zones')
    plt.legend()
    buf_hp_cu = BytesIO()
    plt.savefig(buf_hp_cu, format='png')
    buf_hp_cu.seek(0)
    hp_cu_plot = base64.b64encode(buf_hp_cu.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
    plt.scatter(high_potential_fe['X'], high_potential_fe['Y'], c='blue', label=f'Fe2O3 > {fe_threshold:.2f} %')
    plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('High-Potential Iron Zones')
    plt.legend()
    buf_hp_fe = BytesIO()
    plt.savefig(buf_hp_fe, format='png')
    buf_hp_fe.seek(0)
    hp_fe_plot = base64.b64encode(buf_hp_fe.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
    plt.scatter(high_potential_au['X'], high_potential_au['Y'], c='gold', label=f'Au > {au_threshold:.2f} ppb')
    plt.colorbar(label='Predicted Gold (Au_ppb)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('High-Potential Gold Zones')
    plt.legend()
    buf_hp_au = BytesIO()
    plt.savefig(buf_hp_au, format='png')
    buf_hp_au.seek(0)
    hp_au_plot = base64.b64encode(buf_hp_au.read()).decode('utf-8')
    plt.close()

    return render_template('predict.html', mse_cu=935.87, mse_fe=8.08, mse_au=23.57,
                          orig_cu_plot=orig_cu_plot, orig_fe_plot=orig_fe_plot, orig_au_plot=orig_au_plot,
                          pred_cu_plot=pred_cu_plot, pred_fe_plot=pred_fe_plot, pred_au_plot=pred_au_plot,
                          heatmap_cu_plot=heatmap_cu_plot, heatmap_fe_plot=heatmap_fe_plot, heatmap_au_plot=heatmap_au_plot,
                          hp_cu_plot=hp_cu_plot, hp_fe_plot=hp_fe_plot, hp_au_plot=hp_au_plot,
                          cu_zones=len(high_potential_cu), fe_zones=len(high_potential_fe), au_zones=len(high_potential_au))

@app.route('/download_csv')
def download_csv():
    csv_path = 'mineral_targeting_results_enhanced.csv'
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        abort(404, description="CSV file not found. Please run mineral_targeting1.py to generate it.")
@app.route('/download_report')
def download_report():
    report_path = 'submission_report.txt'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name='submission_report.txt')
    else:
        abort(404, description="Report file not found. Please run mineral_targeting1.py to generate it.")
@app.route('/save_plot/<mineral>/<plot_type>')
def save_plot(mineral, plot_type):
    if mineral == 'cu':
        if plot_type == 'original':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Cu_ppm'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Copper (Cu_ppm)')
            plt.title('Distribution of Copper Concentrations')
            plt.savefig('static/original_cu_scatter.png')
        elif plot_type == 'predicted':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Cu_ppm'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Predicted Copper (Cu_ppm)')
            plt.title('Predicted Copper Concentrations')
            plt.savefig('static/predicted_cu_scatter.png')
        elif plot_type == 'heatmap':
            heatmap, xedges, yedges = np.histogram2d(data_cleaned['X'], data_cleaned['Y'], bins=50, weights=data_cleaned['Predicted_Cu_ppm'])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.figure(figsize=(10, 6))
            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
            plt.colorbar(label='Predicted Copper (Cu_ppm)')
            plt.title('Heatmap of Predicted Copper Concentrations')
            plt.savefig('static/predicted_cu_heatmap.png')
        elif plot_type == 'high_potential':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
            plt.scatter(high_potential_cu['X'], high_potential_cu['Y'], c='red', label=f'Cu > {cu_threshold:.2f} ppm')
            plt.colorbar(label='Predicted Copper (Cu_ppm)')
            plt.title('High-Potential Copper Zones')
            plt.legend()
            plt.savefig('static/high_potential_cu_scatter.png')
    elif mineral == 'fe':
        if plot_type == 'original':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Fe2O3_%'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Iron Oxide (Fe2O3_%)')
            plt.title('Distribution of Iron Oxide Concentrations')
            plt.savefig('static/original_fe_scatter.png')
        elif plot_type == 'predicted':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Fe2O3_%'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
            plt.title('Predicted Iron Oxide Concentrations')
            plt.savefig('static/predicted_fe_scatter.png')
        elif plot_type == 'heatmap':
            heatmap, xedges, yedges = np.histogram2d(data_cleaned['X'], data_cleaned['Y'], bins=50, weights=data_cleaned['Predicted_Fe2O3_%'])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.figure(figsize=(10, 6))
            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
            plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
            plt.title('Heatmap of Predicted Iron Oxide Concentrations')
            plt.savefig('static/predicted_fe_heatmap.png')
        elif plot_type == 'high_potential':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
            plt.scatter(high_potential_fe['X'], high_potential_fe['Y'], c='blue', label=f'Fe2O3 > {fe_threshold:.2f} %')
            plt.colorbar(label='Predicted Iron Oxide (Fe2O3_%)')
            plt.title('High-Potential Iron Zones')
            plt.legend()
            plt.savefig('static/high_potential_fe_scatter.png')
    elif mineral == 'au':
        if plot_type == 'original':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Au_ppb'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Gold (Au_ppb)')
            plt.title('Distribution of Gold Concentrations')
            plt.savefig('static/original_au_scatter.png')
        elif plot_type == 'predicted':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c=data_cleaned['Predicted_Au_ppb'], cmap='viridis', alpha=0.5)
            plt.colorbar(label='Predicted Gold (Au_ppb)')
            plt.title('Predicted Gold Concentrations')
            plt.savefig('static/predicted_au_scatter.png')
        elif plot_type == 'heatmap':
            heatmap, xedges, yedges = np.histogram2d(data_cleaned['X'], data_cleaned['Y'], bins=50, weights=data_cleaned['Predicted_Au_ppb'])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.figure(figsize=(10, 6))
            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
            plt.colorbar(label='Predicted Gold (Au_ppb)')
            plt.title('Heatmap of Predicted Gold Concentrations')
            plt.savefig('static/predicted_au_heatmap.png')
        elif plot_type == 'high_potential':
            plt.figure(figsize=(10, 6))
            plt.scatter(data_cleaned['X'], data_cleaned['Y'], c='gray', alpha=0.1, label='All Data')
            plt.scatter(high_potential_au['X'], high_potential_au['Y'], c='gold', label=f'Au > {au_threshold:.2f} ppb')
            plt.colorbar(label='Predicted Gold (Au_ppb)')
            plt.title('High-Potential Gold Zones')
            plt.legend()
            plt.savefig('static/high_potential_au_scatter.png')
    plt.close()
    return f"Plot for {mineral} ({plot_type}) saved as static/{mineral}_{plot_type}.png"

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000)