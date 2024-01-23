# ConnectTel Customer Churn Prediction

## Overview
This project aims to predict customer churn at ConnectTel, a telecommunications company, using machine learning models. Churn prediction is a crucial task for businesses to identify customers at risk of leaving and take proactive measures to retain them.

## Table of Contents
- [Univariate Analysis](#univariate-analysis)
- [Bivariate Analysis](#bivariate-analysis)
- [Multivariate Analysis](#multivariate-analysis)
- [Correlation Analysis](#correlation-analysis)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Trained](#models-trained)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Univariate Analysis
Explore individual features in the dataset, including customer demographics, services, and contract details, to understand their distributions, summary statistics, and potential outliers. Univariate analysis helps in identifying patterns and trends in each variable.

## Bivariate Analysis
Analyze relationships between pairs of variables, such as the impact of partner status or contract type on churn. Bivariate analysis helps identify correlations or dependencies that may be relevant for predicting customer churn.

## Multivariate Analysis
Examine the simultaneous interactions of multiple variables to uncover complex relationships and patterns. Multivariate analysis provides a holistic view of the interdependencies within the dataset and aids in feature selection for modeling.

## Correlation Analysis
Investigate correlations between features and the target variable (Churn). This step helps identify which features have the strongest influence on customer churn.

## Dataset
The dataset used for this project includes the following features (data dictionary):
- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `TotalCharges`
- `Churn`

The target variable is `Churn`.

## Project Structure
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, and model training.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `data/`: Raw and processed datasets.
- `models/`: Saved machine learning models.

## Models Trained
The following machine learning models were trained and evaluated:
- k-Nearest Neighbors (k-NN)
- Support Vector Classifier (SVC)
- Logistic Regression
- Decision Tree Classifier
- Gaussian Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

## Usage
1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Navigate to the project directory: `cd your-repository`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Explore the Jupyter notebooks in the `notebooks/` directory for data analysis and model training.
5. Use the trained models for predicting customer churn on new datasets.

## Evaluation
Evaluate the models based on metrics such as accuracy, precision, recall, F1-score, and Area Under the Curve (AUC) for ROC curves. The performance may vary depending on the chosen model and dataset characteristics.

## Requirements
- Python 3
- Required Python packages listed in `requirements.txt`

## Contributing
Contributions are welcome! If you find issues or have suggestions for improvements, please create an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
