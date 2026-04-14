# Iris Dataset - Logistic Regression Classification

This project demonstrates a simple logistic regression classification model using the Iris dataset.

## Project Structure
```
AIML Practice/
├── logistic_regression.py    # Main script for logistic regression
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/                     # Data directory
```

## Dependencies
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning library
- numpy: Numerical computing
- matplotlib: Data visualization
- seaborn: Statistical data visualization

## Setup

### Prerequisites
- Python 3.8 or higher installed
- Iris dataset: `C:\Users\uasin\Downloads\iris_dataset.csv`

### Installation Steps

**Step 1:** Configure PowerShell Execution Policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Step 2:** Create Virtual Environment
```powershell
python -m venv venv
```

**Step 3:** Activate Virtual Environment
```powershell
venv\Scripts\Activate.ps1
```
✓ Your prompt should now show `(venv)` prefix:
```
(venv) PS C:\Users\uasin\AIML Practice>
```

**Step 4:** Install Dependencies
```powershell
pip install -r requirements.txt
```

**Step 5:** Run the Script
```powershell
python logistic_regression.py
```

## Usage

Run the logistic regression script:
```powershell
python logistic_regression.py
```

## Dataset
The script uses the Iris dataset from: `C:\Users\uasin\Downloads\iris_dataset.csv`

Ensure the iris_dataset.csv file is in the Downloads folder.

## Model Details

**Algorithm**: Logistic Regression
- **Purpose**: Multi-class classification
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Target**: 3 iris species classes
- **Train-Test Split**: 80-20
- **Feature Scaling**: StandardScaler (normalized)
- **Max Iterations**: 1000

## Output

The script generates:
1. **Model Accuracy**: Overall accuracy on test set
2. **Classification Report**: Precision, Recall, F1-Score per class
3. **Confusion Matrix**: True positives, false positives, etc.
4. **Visualization**: confusion_matrix.png (saved locally)
5. **Model Coefficients**: Feature importance/weights

## Example Output
```
Dataset shape: (150, 5)
Training set size: 120
Testing set size: 30

==================================================
MODEL PERFORMANCE
==================================================
Accuracy: 0.9333

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

## Tips & Improvements

- Experiment with different train-test split ratios
- Try different random seeds for reproducibility
- Use cross-validation for more robust evaluation
- Tune hyperparameters (max_iter, C, solver)
- Implement feature engineering for better performance
- Use ensemble methods like Random Forest or SVM for comparison

## References
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)