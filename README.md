# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details. The project includes data preprocessing, exploratory data cleaning, model comparison, hyperparameter tuning, model serialization, and a desktop GUI built with Tkinter.

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications every day. Evaluating each application manually is time-consuming and prone to human error.

This project automates the loan approval process using supervised machine learning algorithms. Multiple classification models are trained and compared using cross-validation. The best-performing model is optimized using hyperparameter tuning and deployed through a simple GUI application.

---

## 🚀 Features

- Data preprocessing and cleaning
- Missing value handling
- Categorical feature encoding
- Feature scaling using StandardScaler
- Multiple Machine Learning algorithms
- 5-Fold Cross Validation
- Hyperparameter tuning using RandomizedSearchCV
- Best model selection
- Model saving using Joblib
- Desktop GUI using Tkinter
- Instant loan approval prediction

---

## 📂 Dataset

The project uses the **Loan Prediction Dataset** containing applicant information such as:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (Target Variable)

---

## 🛠 Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns (`Loan_ID`)
- Handled missing values
- Filled missing categorical values using Mode
- Replaced "3+" dependents with 4
- Converted categorical variables into numerical values using mapping
- Standardized numerical features using **StandardScaler**

---

## 🤖 Machine Learning Models Used

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

---

## 📊 Model Evaluation

Each model was evaluated using:

- Train-Test Split
- Accuracy Score
- 5-Fold Cross Validation

Example:

```python
cross_val_score(model, X, y, cv=5)
```

---

## ⚙ Hyperparameter Tuning

RandomizedSearchCV was used to optimize the following models:

### Logistic Regression

- C
- Solver

### Support Vector Machine

- C
- Kernel

### Random Forest

- Number of Trees
- Maximum Depth
- Maximum Features
- Minimum Samples Split
- Minimum Samples Leaf

Example:

```python
RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    cv=5,
    n_iter=20
)
```

---

## 🏆 Best Model

After comparing all models and tuning hyperparameters, the **Random Forest Classifier** produced the best performance and was selected as the final model.

The trained model was saved using Joblib.

```python
joblib.dump(model, "loan_status_predict")
```

---

## 💻 GUI Application

The project includes a **Tkinter-based desktop application** where users can enter applicant details and instantly receive a prediction.

Input Fields:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

Output:

- ✅ Loan Approved
- ❌ Loan Not Approved

---

## 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── loan_prediction.csv
├── loan_prediction.py
├── loan_status_predict
├── README.md
│
└── screenshots/
```

---

## 📦 Libraries Used

```text
pandas
numpy
scikit-learn
joblib
tkinter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Loan-Approval-Prediction.git
```

### Navigate to the project

```bash
cd Loan-Approval-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy scikit-learn joblib
```

### Run the project

```bash
python loan_prediction.py
```

The Tkinter GUI will open where you can enter applicant information and predict the loan approval status.

---

## 📈 Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Handling Missing Values
    │
    ▼
Feature Encoding
    │
    ▼
Feature Scaling
    │
    ▼
Train-Test Split
    │
    ▼
Model Training
    │
    ▼
5-Fold Cross Validation
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Best Model Selection
    │
    ▼
Save Model
    │
    ▼
Tkinter GUI
    │
    ▼
Loan Prediction
```

---

## 📸 Screenshots

You can add screenshots of:

- Dataset
- Model comparison
- Cross-validation results
- Hyperparameter tuning
- Tkinter GUI
- Loan prediction output

Example:

```
screenshots/
    gui.png
    prediction.png
```

---

## 🎯 Future Improvements

- Develop a Flask or Django web application
- Deploy on Render, Railway, or Heroku
- Add probability scores for predictions
- Improve GUI design
- Implement explainable AI using SHAP or LIME
- Integrate with a real-world banking database

---

## 👨‍💻 Author

**Ujjwal Mittal**

If you found this project useful, consider giving it a ⭐ on GitHub!
