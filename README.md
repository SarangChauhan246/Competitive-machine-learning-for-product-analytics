## Predictive Analysis for Product Conversion

### Overview
This project is designed to predict product conversions based on user interactions and product data. It leverages various machine learning algorithms such as Logistic Regression, K-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost to analyze and predict conversion rates.

### Data
The dataset used in this project contains user interactions on a website, captured in a CSV file named `product.csv`. The columns in the dataset are:

- `order_id`: Unique identifier for each order
- `user_id`: Unique identifier for each user
- `page_id`: Unique identifier for each page
- `product`: Category of the product
- `site_version`: Version of the site (desktop or mobile)
- `time`: Timestamp of the interaction
- `title`: Type of interaction (e.g., banner_click, banner_show)
- `target`: Binary target variable indicating whether the interaction led to a conversion (1) or not (0)

### Data Preprocessing
The data preprocessing steps include:

1. **Handling Missing Values**: Filling NaN values with 0 for `order_id` and `page_id`.
2. **Feature Engineering**:
   - Aggregating user interactions to derive features such as the number of conversions, whether a banner was clicked, and if it was the first conversion.
   - Extracting time-related features like hour, day, weekday, and year-month from the timestamp.
   - Converting categorical features to dummy variables.
3. **Undersampling**: Balancing the dataset by undersampling the majority class (non-conversions) to match the number of conversion records.

### Feature Importance
Feature importance is determined using three methods:

1. **SelectKBest**: Selecting top features based on ANOVA F-value.
2. **Logistic Regression with L2 Regularization**: Assessing feature importance using coefficients.
3. **XGBoost Classifier**: Using feature importances from the trained model.

### Model Training and Evaluation
The following models are trained and evaluated on the preprocessed data:

1. **Bernoulli Naive Bayes**
2. **K-Nearest Neighbors**
3. **Random Forest Classifier**
4. **Logistic Regression**
5. **XGBoost Classifier**

The models are evaluated using metrics such as accuracy and ROC AUC score.

### Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV for the Random Forest and Logistic Regression models to find the optimal parameters that yield the best performance.

### Clustering
K-Means clustering is applied to identify potential patterns in the data. The dataset is scaled, and clusters are created to segment users based on their interactions and features.

### Results
The results of the models, including feature importances, evaluation metrics, and clustering outcomes, are summarized to provide insights into the key factors influencing product conversions and the performance of different predictive models.

### Usage
To use this project:

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

2. **Run the Python Script**:
   Ensure the dataset `product.csv` is in the same directory as the script. Execute the script to preprocess the data, train the models, evaluate their performance, and obtain insights from feature importance and clustering analysis.

3. **Explore Results**:
   Review the printed evaluation metrics and feature importance scores to understand the performance of each model and the significance of each feature.

### Conclusion
This project demonstrates a comprehensive approach to predictive analysis using machine learning, providing valuable insights into user interactions and product conversions. The results can be used to optimize marketing strategies, enhance user experience, and ultimately improve conversion rates.
