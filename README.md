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

### 📊 Example Model Output

```
Accuracy: 0.94
Top Features:
- grid
- driver_points_last5
- driver_avg_finish_last5
- constructor_points_last5
```

### 🔮 Next Steps
- Add predict_next_race.py to forecast winners for upcoming races
- Experiment with more ML algorithms like CatBoost
- Deploy as a web app using Streamlit

### 📜 License
This project is licensed under the MIT License.

### 🙌 Acknowledgements
Jolpica F1 API for race data
Scikit-learn for model building