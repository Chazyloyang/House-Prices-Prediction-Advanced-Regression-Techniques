# House-Prices-Prediction-Advanced-Regression-Techniques
# Project Overview
This project is a complete end-to-end machine learning pipeline for the Kaggle House Prices competition. The goal is to predict the final sale price of homes in Ames, Iowa, based on a comprehensive set of features. The process includes data cleaning, exploratory data analysis (EDA), feature engineering, model selection, training, and generating a submission file.

# Competition
This project was developed to compete in the Kaggle House Prices: Advanced Regression Techniques competition.

Competition Link: (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

# Files
train.csv: The training dataset, including the target variable SalePrice.

test.csv: The test dataset, without the SalePrice variable.

submission.csv: The final output file with the predicted SalePrice for the test data, formatted for submission to the competition.

train_xgboost_model.py: The main Python script that runs the entire pipeline, from data loading to submission file generation.

data_visualization.py: A supporting script for in-depth data visualization and correlation analysis.

# Methodology
1. Data Cleaning & Preprocessing
The initial raw data contained missing values and categorical features that needed to be prepared for a machine learning model.

Missing categorical data (e.g., PoolQC, Alley) was imputed with a 'None' value, indicating the absence of that feature.

Missing numerical data (e.g., LotFrontage) was imputed with the median value of its neighborhood, while other missing numerical values were filled with 0.

2. Feature Engineering & Transformation
Several new features were created to help the model learn more effectively, and existing features were transformed to better fit the model's assumptions.

A log transformation (np.log1p) was applied to the target variable (SalePrice) to normalize its highly skewed distribution. This step is crucial because the competition's evaluation metric is the Root-Mean-Squared-Error (RMSE) on the log-transformed prices.

Other skewed numerical features, such as GrLivArea and LotArea, were also log-transformed to linearize their relationship with the target variable.

A new feature, YearsSinceRemod, was created by subtracting YearRemodAdd from YrSold, providing a more direct measure of a home's age after its last major renovation.

Original, untransformed columns were dropped from the dataset to avoid multicollinearity.

3. Exploratory Data Analysis (EDA)
Before modeling, the data was visualized to confirm that the cleaning and transformations were successful.

Histograms showed the distribution of SalePrice before and after the log transformation, visually confirming a shift from a skewed distribution to a more normal, bell-curve shape.

Scatter plots demonstrated how the transformation helped to linearize the relationship between features like GrLivArea and SalePrice, which is a key assumption for many models.

A correlation heatmap was generated to quickly identify the top 15 features most strongly correlated with log_SalePrice.

4. Model Selection & Training
For this regression task, several models could have been used, but XGBoost (Extreme Gradient Boosting) was selected as the final model.

Reasoning for XGBoost: XGBoost is an ensemble model known for its high performance on tabular data. Unlike simpler models like Linear Regression, it can effectively capture the complex, non-linear relationships and feature interactions present in the dataset, leading to a much more accurate prediction.

The model was trained on the pre-processed and one-hot encoded training data.

5. Prediction & Submission
The trained XGBoost model was used to predict the log_SalePrice for the test dataset.

The predictions were then inverse-transformed using np.expm1 to convert them back to the original SalePrice scale.

Finally, a submission.csv file was generated, containing the Id and SalePrice predictions in the required format for the competition.

# How to Run the Project
Clone this repository to your local machine.

Download the datasets (train.csv and test.csv) from the Kaggle competition page and place them in the root directory of this project.

Install the required libraries using pip:

pip install pandas numpy scikit-learn xgboost

