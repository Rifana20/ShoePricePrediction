# 👟 Shoe Price Predictor

![Screenshot (761)](https://github.com/user-attachments/assets/e2262c5f-5974-4fa2-b2c4-ca85cb648d26)
![Screenshot (762)](https://github.com/user-attachments/assets/858392d3-5117-4c4f-a428-d94f81da69d6)


A machine learning-powered web application that predicts the price of a shoe based on features like brand, type, gender, material, and size. Built using **Python**, **Scikit-Learn**, and **Streamlit**.

🔗 **Live App:** [https://shoepriceprediction-2dtxlpjaxutpuhvbcyhkmr.streamlit.app/](https://shoepriceprediction-2dtxlpjaxutpuhvbcyhkmr.streamlit.app/)

---

## 📦 Features

- Predicts shoe price instantly using a trained machine learning model
- Intuitive UI built with Streamlit
- Dropdowns for Brand, Gender, Type, Material
- Slider for Size input
- Stylish background and themed headings

---

## 🧠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Random Forest Regressor
- Joblib

---
### Model Performance:
| Metric | Value |
|--------|-------|
| R² Score | **0.777** |
| MAE | **$11.76** |
| RMSE | **$18.22** |

The model is saved as `shoe_price_model.pkl` using `joblib`.
 My model can explain about 78% of the shoe price variation based on the input features.
 Mean Absolute Error (MAE): 11.76 The model typically makes an error of about 12 dollars per prediction.
Root Mean Squared Error (RMSE): 18.22 While most predictions are ~12 dollars off, a few could be off by 18 or more.

---


## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Rifana20/ShoePricePrediction.git
cd ShoePricePrediction

- 2.Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # For Windows
# OR
source venv/bin/activate  # For Mac/Linux

- 3.Install Dependencies
pip install -r requirements.txt

-4.Run the App
streamlit run app.py


