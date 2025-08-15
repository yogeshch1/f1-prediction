# 🏎️ F1 Race Winner Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

Predict upcoming Formula 1 race winners using historical race data and machine learning models like **Random Forest**, **XGBoost**, and **LightGBM**.

---

## 📌 Features
✅ Fetch latest and historical F1 data from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1)  
✅ Clean and process race results into structured datasets  
✅ Engineer features for drivers and constructors  
✅ Train and evaluate machine learning models  
✅ Save trained models for future predictions  

---

## 📂 Project Structure
```
f1-prediction/
│
├── src/                   # Source scripts
│   ├── collect_f1_history.py
│   ├── prepare_features.py
│   ├── train_model.py
│   ├── get_latest_f1_winner.py
│
├── data/                  # Raw/processed datasets (ignored in GitHub)
├── models/                # Saved models (ignored in GitHub)
├── .gitignore
├── README.md
├── requirements.txt
└── venv/                  # Virtual environment (ignored)
```

---

## 🚀 How to Run

### 1️⃣ Clone this repository
```
git clone https://github.com/yogeshch1/f1-prediction.git
cd f1-prediction
```

### 2️⃣ Create virtual environment
```
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Collect historical F1 data
```
python src/collect_f1_history.py
```

### 5️⃣ Prepare features
```
python src/prepare_features.py
```

### 6️⃣ Train the model
```
python src/train_model.py
```

## 📊 Model Performance (Latest Run)

**Training seasons:** 2018–2024  
**Validation season:** 2025  

- **Train rows:** 2,979  
- **Validation rows:** 279  
- **Features used:**  
  `['grid', 'driver_avg_finish_last5', 'driver_wins_last5', 'driver_podiums_last5', 'driver_avg_grid_last5', 'driver_points_last5', 'constructor_avg_finish_last5', 'constructor_points_last5', 'driver_track_avg_finish']`  

### 🏁 Validation Results

| Metric       | 0 (Not Winner) | 1 (Winner) | Macro Avg | Weighted Avg |
|--------------|---------------|-----------|-----------|--------------|
| **Precision** | 0.98          | 0.44      | 0.71      | 0.95         |
| **Recall**    | 0.96          | 0.57      | 0.77      | 0.94         |
| **F1-score**  | 0.97          | 0.50      | 0.73      | 0.95         |
| **Accuracy**  | -             | -         | **0.94**  | **0.94**     |
| **Support**   | 265           | 14        | 279       | 279          |

---

### 🔝 Top Features (by importance)

1. `grid` – 0.248  
2. `driver_points_last5` – 0.197  
3. `driver_avg_finish_last5` – 0.150  
4. `driver_avg_grid_last5` – 0.126  
5. `constructor_points_last5` – 0.105  
6. `driver_podiums_last5` – 0.074  
7. `constructor_avg_finish_last5` – 0.044  
8. `driver_track_avg_finish` – 0.032  
9. `driver_wins_last5` – 0.023  

---

**Model file:** `models/f1_winner_model_clean.pkl`  

---

### 📖 Interpretation

- **Accuracy (94%)**: The model correctly predicts whether a driver will win in 94% of cases overall.  
- **Winner prediction is harder**: Precision for winners is 0.44, meaning when the model predicts a win, it’s correct 44% of the time. Recall is 0.57, meaning it catches 57% of actual winners. This is expected since predicting the single race winner is a much tougher task than predicting non-winners.  
- **Grid position matters most**: Starting position (`grid`) is the most important feature, followed by recent driver form (`driver_points_last5`) and recent average finish (`driver_avg_finish_last5`).  
- **Constructor performance**: Recent constructor form also plays a notable role (`constructor_points_last5`, `constructor_avg_finish_last5`).  

**Takeaway:** The model is strong at predicting the overall race outcome distribution but still misses some winners. Adding more pre-race variables (e.g., weather, qualifying gaps, track-specific performance) could improve winner recall.


### 🔮 Next Steps
- Add predict_next_race.py to forecast winners for upcoming races
- Experiment with more ML algorithms like CatBoost
- Deploy as a web app using Streamlit

### 📜 License
This project is licensed under the MIT License.

### 🙌 Acknowledgements
[Jolpica F1 API](https://github.com/jolpica/jolpica-f1) for race data  
[Scikit-learn](https://scikit-learn.org/stable/) for model building