import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# function to convert strings to floats
def to_float(value):
    try:
        return float(value)
    except ValueError:
        # Extracting number if it's a string and convert it to float
        if isinstance(value, str) and '>' in value:
            return float(value[1:])
        # If other non-numeric value, return NaN
        return np.nan

# Loading the dataset
data_path = '/Users/aashrithamachiraju/Desktop/ovarian-cancer-detection/data/Infer OC/Supplementary data 1.xlsx'
df = pd.read_excel(data_path)

# Apply the to_float function and imputing median values
for col in df.columns:
    if col not in ['SUBJECT_ID', 'TYPE']:
        df[col] = df[col].apply(to_float)
        df[col] = df[col].fillna(df[col].median())

# Ensuring 'TYPE' is recognized as numeric and has more than one unique value
df['TYPE'] = pd.to_numeric(df['TYPE'], errors='coerce')
if df['TYPE'].nunique() < 2:
    raise ValueError("The target variable 'TYPE' has less than two unique values.")

# Correlation matrix
corr_matrix = df.corr()

# Plotting a heatmap of the correlations with respect to the target variable
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix including TYPE')
plt.show()

# Selecting features with high absolute correlation to the target variable
threshold = 0.3
high_corr_features = corr_matrix.index[abs(corr_matrix['TYPE']) > threshold].tolist()
high_corr_features.remove('TYPE')

# Scaling the features before calculating VIF
X = df[high_corr_features]  # Features DataFrame
X_scaled = StandardScaler().fit_transform(X)

# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

# Displaying the VIF for each feature
print(vif_data.sort_values('VIF', ascending=False))

# Removing features with a high VIF
vif_threshold = 5  # This is a commonly used threshold
features_to_use = vif_data[vif_data["VIF"] < vif_threshold]["Feature"].tolist()

# Creating a new DataFrame with the reduced set of features
reduced_df = df[features_to_use + ['TYPE']]

# Printing the remaining features
print("Features to be used in the model:", features_to_use)

# This is the output of the above data wrangling and the features to be used:
# Features to be used in the model: ['SUBJECT_ID', 'Age', 'ALB', 'CA125', 'HE4', 'LYM%', 'Menopause', 'NEU']

# Reformat the title of LYM%
reduced_df = reduced_df.rename(columns={'LYM%': 'LYM_percent'})

# Establish the list of variables to loop for different logistic regression tests.
variables = ['Age', 'ALB', 'LYM_percent', 'Menopause', 'NEU']

for var in variables:
    formula = f'TYPE ~ {var}'
    model = smf.logit(formula, data=reduced_df)
    result = model.fit()
    print(f"Logistic Regression Results for {var}:")
    print(result.summary())
    print("\n")

#For individual tests for CA125 and HE4, these variables need to undergo log transformation prior to testing.
log_variables = ['CA125', 'HE4']

for logV in log_variables:
    formula = f'TYPE ~ np.log10({logV})'
    model = smf.logit(formula, data=reduced_df)
    result = model.fit()
    print(f"Logistic Regression Results for {logV}:")
    print(result.summary())
    print("\n")


