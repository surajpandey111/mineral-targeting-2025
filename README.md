Mineral Targeting 2025: AI-Driven Solution for Critical Mineral Exploration
Overview
"Mineral Targeting 2025" is an AI-driven solution developed for the IndiaAI Hackathon on Mineral Targeting, aimed at identifying high-potential zones for critical minerals (Copper, Iron, Gold) across a 39,000 sq. km area in Karnataka and Andhra Pradesh, India. The project leverages multi-parametric geoscience data from the Geological Survey of India (GSI) AIKosh platform, employing advanced machine learning and deep learning techniques to predict concealed and deep-seated ore bodies. Two approaches were implemented:

First Approach: Utilized three datasets (NGCM, ASTER, Aerogeophysical) and seven ML/AI models, including STGNN, CNN, QIENN, Linear Regression, Tuned XGBoost, Tuned Stacking Ensemble, and Weighted Average Ensemble.
Second Approach: Used a single NGCM dataset with a Random Forest Regressor (RFR) model, deployed via Flask and a static React site.

Both approaches outperformed a 20% random baseline by fourfold, identifying 501 high-potential locations per mineral, with results visualized through 3D/2D scatter plots, heatmaps, and anomaly detection.
Repository Structure

Main Repository: https://github.com/surajpandey111/mineral-targeting-2025 Contains code, datasets, models, and visualizations for the First Approach.
RFR-NGCM Repository: https://github.com/surajpandey111/mineral-targeting-2025/tree/main/RFRngcm Contains code and deployment for the Second Approach.
Static Site Repository: https://github.com/surajpandey111/mineral-targeting-static Hosts the static React site deployment.

Methodology
First Approach: Multi-Dataset with Seven ML/AI Models

Data Preparation:
Datasets: National Geochemical Mapping (NGCM) dataset (NGCM-Stream-Sediment-Analysis-Updated.xlsx, 10,004 entries, 73 columns), ASTER, and Aerogeophysical data (magnetic anomalies).
Processing: Loaded with pandas, cleaned using IQR for outliers, normalized with StandardScaler, and unified into processed_data.pkl for efficiency.
Features: X, Y, Cu_ppm, Fe2O3_%, Au_ppb, Ni_ppm, Zn_ppm, Pb_ppm, raster bands, and magnetic anomalies.


Model Development:
Split data into 80% training and 20% testing (random_state=42).
Models:
STGNN: 3 layers (Input → GCNConv1 → GCNConv2), 5-nearest-neighbor graph, MSE 211.72.
CNN: 5 layers (Input → Conv1d_1 → Conv1d_2 → FC1 → FC2), MSE 10.54.
QIENN: Optimized STGNN and CNN weights over 10 generations, MSE 8.0070.
Linear Regression: Baseline model.
Tuned XGBoost: n_estimators=200, learning_rate=0.05, MSE 211.7205.
Tuned Stacking Ensemble: Meta-model with XGBoost, MSE 7.8744.
Weighted Average Ensemble: Weighted combination of all models (inverse MSE), MSE 10.3385.


Feature Importance: Ni_ppm_scaled (35%), Cu_ppm (25%).
Cross-Validation: 5-fold stratification, 95% accuracy.


Visualization:
3D scatter plots (go.Scatter3d), 2D scatter plots, heatmaps (plt.tricontourf), high-potential zones (top 5% quantile), correlation heatmaps (Seaborn), and anomaly detection (z-scores >3).
Outputs: mineral_targeting_results_enhanced.csv, various PNG files (e.g., Heatmap_Tuned_Stacking_Ensemble_Predicted_Copper_Concentration.png).



Second Approach: Single Dataset with Random Forest Regressor

Data Preparation:
Dataset: NGCM (NGCM-Stream-Sediment-Analysis-Updated.xlsx).
Processing: Cleaned with IQR, normalized with StandardScaler, validated with data_cleaned.info().


Model Development:
RFR: 200 estimators, max depth 10, MSE values: Copper (935.87), Iron (8.08), Gold (23.57).
Feature Importance: Ni_ppm_scaled (35%), Cu_ppm (25%).
Cross-Validation: 5-fold stratification, 88% accuracy.


Visualization:
Scatter plots (plt.scatter), heatmaps (np.histogram2d), high-potential zones (top 5% quantile).
Outputs: mineral_targeting_results_enhanced.csv, PNG files (e.g., CopperConcentration.png).


Deployment:
Flask App: https://mineral-targeting-2025.onrender.com/ for dynamic predictions.
Static React Site: https://mineral-targeting-static.onrender.com/ for scalable access.



Results

First Approach:
Best Model: Tuned Stacking Ensemble (MSE 7.8744).
High-Potential Zones: 501 per mineral (Copper, Iron, Gold).
Visualizations: 3D/2D plots, heatmaps, correlation heatmaps, anomaly detection.


Second Approach:
RFR MSE: Copper (935.87), Iron (8.08), Gold (23.57).
High-Potential Zones: 501 per mineral.
Deployment: Flask app and static site, accessible with downloadable results (mineral_targeting_results_enhanced.csv, submission_report.txt).


Both approaches outperformed a 20% random baseline by fourfold, aligning with IndiaAI’s mission for AI-driven mineral exploration.

Installation

Clone the repository:git clone https://github.com/surajpandey111/mineral-targeting-2025.git
cd mineral-targeting-2025


Install dependencies:pip install -r requirements.txt

Requirements include: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, torch, torch-geometric, rasterio, scipy, flask.
For the static site:git clone https://github.com/surajpandey111/mineral-targeting-static.git
cd mineral-targeting-static
npm install
npm run build



Usage

First Approach:
Run data preprocessing:python scripts/preprocess_data.py


Train models:python scripts/train_models.py


Generate visualizations:python scripts/visualize_results.py




Second Approach:
Run RFR training:cd RFRngcm
python train_rfr.py


Launch Flask app locally:python app.py


Access the static site after building:npm start





Deployment

Flask App (Second Approach): https://mineral-targeting-2025.onrender.com/
Static Site (Second Approach): https://mineral-targeting-static.onrender.com/

Future Work

Enhance datasets with GSI’s Geochronology, Ground Gravity, and Borehole data.
Refine neural networks (STGNN, CNN, QIENN) with deeper architectures and advanced optimization.
Develop 3D depth models for concealed ore bodies.
Validate predictions through GSI field surveys.
Expand deployment with real-time prediction capabilities.

References

Geological Survey of India (GSI). (n.d.). AIKosh Platform Data. Retrieved from https://aikosh.gsi.gov.in/.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241-259.
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
LeCun, Y., et al. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278-2324.
Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. International Conference on Learning Representations (ICLR).
Biamonte, J., et al. (2017). Quantum Machine Learning. Nature, 549(7671), 195-202.
Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32.

Contact

Authors: Suraj Kumar Pandey, Prof.B.K. Tripathi
Email: worldforensic@gmail.com
Phone: +917488723028

Join the IndiaAI Hackathon on Mineral Targeting (deadline: May 12, 2025) to contribute to AI-driven mineral discovery! For queries, contact: fellow3.gpai-india@meity.gov.in.
