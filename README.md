# 🏠 House Price Prediction
## 🚀 Run the App
pip install streamlit scikit-learn pandas numpy
streamlit run f1.py

Then open: http://localhost:8501

---

## 📊 Model Performance

| Metric     | Value             |
|------------|-------------------|
| Algorithm  | Linear Regression |
| R² Score   | 0.77              |
| MSE        | ~18               |
| Scaler     | StandardScaler    |
| Test Size  | 20%               |
| Dataset    | Boston Housing    |

---

## ✨ Features

- 13 interactive sliders for real-time input
- Instant house price prediction in dollars
- Model info sidebar with R² and MSE
- Pre-trained model loaded via Pickle
- Clean beginner-friendly UI

---

## 🏘️ Input Features

| Feature  | Description                    |
|----------|--------------------------------|
| CRIM     | Crime rate per capita          |
| ZN       | % land for large homes         |
| INDUS    | % non-retail business acres    |
| CHAS     | Near Charles River? (0/1)      |
| NOX      | Nitric oxide concentration     |
| RM       | Avg rooms per dwelling         |
| AGE      | % homes built before 1940      |
| DIS      | Distance to employment centres |
| RAD      | Highway accessibility index    |
| TAX      | Property tax rate              |
| PTRATIO  | Pupil-teacher ratio            |
| B        | Diversity index                |
| LSTAT    | % lower-status population      |

---

## 📁 Project Structure

house-price-prediction/
├── f1.py                        # Streamlit web app
├── model.pkl                    # Trained ML model
├── scaler.pkl                   # Fitted StandardScaler
├── House_price_prediction.ipynb # Training notebook
└── README.md

---

## 🛠️ Tech Stack

- Language   → Python 3
- Web App    → Streamlit
- ML Model   → Scikit-learn LinearRegression
- Scaling    → StandardScaler
- Data       → Pandas + NumPy
- Deployment → Pickle

---

## 🧠 What I Learned

- Data cleaning and handling missing values
- Outlier removal using IQR method
- Feature scaling with StandardScaler
- Training and evaluating Linear Regression
- Saving and loading models with Pickle
- Building a web app with Streamlit
- End-to-end ML deployment pipeline


## 👨‍💻 Author

Fawad Ahmad

LinkedIn → (https://www.linkedin.com/in/fawad-ahmad-bilal-78890236b/)


⭐ If this project helped you, please give it a star!
