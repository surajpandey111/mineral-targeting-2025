# Mineral Targeting 2025

This project is a submission for the **IndiaAI Hackathon on Mineral Targeting 2025**, a collaboration between IndiaAI (under the Digital India Corporation, MeitY) and the Geological Survey of India (GSI). Developed by **Suraj Kumar Pandey** (Developer/Scientist/Student), the initiative leverages Artificial Intelligence (AI) and Machine Learning (ML) to identify new potential areas for exploration of critical minerals (e.g., REE, Ni-PGE, Copper) and other commodities (e.g., Diamond, Iron, Manganese, Gold) within a 39,000 sq. km area in Karnataka and Andhra Pradesh, India.

## Project Overview
- **Theme**: Identification of concealed and deep-seated mineralized bodies with depth modeling, using AI/ML algorithms for data cleaning, integration, modeling, validation, and generating mineral predictive maps.
- **Objective**: Enhance the effectiveness of critical mineral exploration by integrating multi-parametric geoscience data (geology, geophysics, geochemistry, remote sensing, borehole data, etc.).
- **Live Demo**: [https://mineral-targeting-2025.onrender.com/](https://mineral-targeting-2025.onrender.com/)
- **Live Demo**: [React js static page of mineral-targeting-result](https://mineral-targeting-static.onrender.com/)

## Features
- Predicts Copper (Cu_ppm), Iron Oxide (Fe2O3_%), and Gold (Au_ppb) concentrations using Random Forest Regressor.
- Generates scatter plots, heatmaps, and high-potential zone visualizations.
- Web app with downloadable CSV results (`mineral_targeting_results_enhanced.csv`) and submission report (`submission_report.txt`).
- Attractive welcome page with developer details and hackathon info.

## Installation
1. Clone the repository: `git clone https://github.com/surajpandey111/mineral-targeting-2025.git`
2. Install dependencies: `pip install -r web_app/requirements.txt`
3. Navigate to web_app: `cd web_app`
4. Generate precomputed data: `python mineral_targeting1.py` (if needed)
5. Run locally: `python app.py`

## Usage
- Visit `http://localhost:5000/` for the welcome page.
- Visit `http://localhost:5000/predict` for predictions and visualizations.
- Download results via `/download_csv` and report via `/download_report`.
- Save plots via `/save_plot/<mineral>/<plot_type>`.

## Deployment
- **Hosted on Render**: [https://mineral-targeting-2025.onrender.com/](https://mineral-targeting-2025.onrender.com/)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Root Directory**: `/web_app`
## Static Page Deployemnt
- **Live Demo**: [React js static page of mineral-targeting-result](https://mineral-targeting-static.onrender.com/)
- **React Code repositories of static deployement**:[React js static page of mineral-targeting-result](https://github.com/surajpandey111/mineral-targeting-static)
## Datasets
This project utilizes the following datasets provided by GSI on AIKosh:
- Multi-Layer Geological Map of Karnataka and Andhra Pradesh (25K & 50K Scale)
- Geochronology, Geomorphology, and Lineament Features Maps
- Geochemical Data Points (NGCM)
- Aerogeophysical Magnetic and Spectrometric Data
- Ground Gravity Data
- Mineral Exploration Blocks and ASTER Mineral Maps
- Technical reports on mineral exploration
- [Access Datasets Here](https://aikosh.indiaai.gov.in/home)

Additional data: `NGCM-Stream-Sediment-Analysis-Updated.xlsx` (processed and included).

## Credits
- **Developed by**: Suraj Kumar Pandey
- **Collaboration**: IndiaAI and Geological Survey of India (GSI)
- **License**: MIT License

## Submission for IndiaAI Hackathon 2025
- **Source Code URL**: [https://github.com/surajpandey111/mineral-targeting-2025](https://github.com/surajpandey111/mineral-targeting-2025)
- **Live URL**: [https://mineral-targeting-2025.onrender.com/](https://mineral-targeting-2025.onrender.com/)
- - **Live Demo**: [React js static page of mineral-targeting-result](https://mineral-targeting-static.onrender.com/)
- **Project Report**: Included as `submission_report.txt` in the repository.
- **Submission Deadline**: 12th May 2025
- **Contact**: Email `fellow3.gpai-india@meity.gov.in` or `sudheer.reddy@meity.gov.in` with subject "Queries for IndiaAI Hackathon on Mineral Targeting - Suraj Kumar Pandey "

## Prizes
- First Prize: INR 10 Lakhs
- Second Prize: INR 7 Lakhs
- Third Prize: INR 5 Lakhs
- Special Prize: INR 5 Lakhs for All-Women Teams (if no women team in top 3)
