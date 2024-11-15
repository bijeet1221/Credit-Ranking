```markdown
# Credit Score Prediction Project

## Overview

This project aims to build a machine learning model to predict **credit score categories** (`good`, `standard`, `poor`) based on customer data. The model leverages various supervised learning algorithms, and the **Gradient Boosting (SGB)** model achieved the highest accuracy during experimentation.

---

## Features and Dataset

The dataset contains 28 features and 80,000 rows, providing details such as:

- **ID**: Unique customer identifier.
- **Customer_ID**: Secondary unique identifier.
- **Age**: Age of the customer.
- **Income_Annual**: Annual income of the customer.
- **Base_Salary_PerMonth**: Fixed monthly salary.
- **Credit_Limit**: Assigned credit limit.
- **Credit_Score**: Target variable categorized as `good`, `standard`, or `poor`.

---

## Models Used

The following machine learning models were explored during the development process:

1. **Decision Tree Classifier**
   - A simple yet interpretable model that forms the baseline for comparison.
   - Achieved moderate accuracy.

2. **AdaBoost Classifier**
   - An ensemble method that improves performance by combining weak learners iteratively.
   - Showed improved accuracy over the Decision Tree.

3. **Random Forest Classifier**
   - Utilizes multiple decision trees for robust and stable predictions.
   - Balanced performance but not the highest.

4. **XGBoost Classifier**
   - A powerful gradient boosting framework optimized for speed and performance.
   - Performed well but slightly lagged behind SGB.

5. **Gradient Boosting Classifier (SGB)**
   - The top-performing model with the highest accuracy.
   - Offers fine-tuned control over learning parameters for optimal results.

---

## Results

After hyperparameter tuning using **Optuna** for XGBoost and **BayesSearchCV** for Gradient Boosting and Random Forest, the **Gradient Boosting Classifier (SGB)** emerged as the best model:

- **Highest Accuracy**: Achieved using **SGB** during testing and validation.

---

## Technology Stack

- **Programming Language**: Python
- **Libraries**:
  - Scikit-learn
  - XGBoost
  - Optuna
  - Scikit-optimize (BayesSearchCV)
  - Pandas, NumPy
  - Matplotlib, Seaborn (for visualization)

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-score-prediction.git
   cd credit-score-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script to train models:
   ```bash
   python train_models.py
   ```

4. Evaluate the performance:
   ```bash
   python evaluate_models.py
   ```

5. (Optional) Perform hyperparameter tuning using Optuna:
   ```bash
   python hyperparameter_tuning.py
   ```

---

## Key Insights

- Ensemble models like **Gradient Boosting** and **XGBoost** outperform single decision tree models for complex datasets.
- Proper hyperparameter tuning significantly enhances model performance.
- The **Gradient Boosting Classifier** strikes the best balance between accuracy and interpretability.

---

## Future Work

- Expand the feature set for better predictions.
- Deploy the best-performing model as a web service or API.
- Automate the pipeline for end-to-end credit scoring.

---

Feel free to explore, contribute, or report any issues! 🎉

---

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
