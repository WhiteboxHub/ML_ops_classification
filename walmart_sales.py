




# import os
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import dvc.api
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # Set up warnings and plotting
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['figure.figsize'] = [10, 6]

# # Load dataset using DVC
# data_path = 'data/Walmart.csv'
# with dvc.api.open(data_path, mode='r') as f:
#     df = pd.read_csv(f)

# # Data preprocessing
# # Fix the date format issue
# df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Use the correct format
# df['weekday'] = df['Date'].dt.weekday
# df['month'] = df['Date'].dt.month
# df['year'] = df['Date'].dt.year
# df.drop(['Date'], axis=1, inplace=True)

# # Define target and features
# target = 'Weekly_Sales'
# features = [i for i in df.columns if i not in [target]]

# # Display the dataset
# print("Dataset Head:")
# print(df.head())

# # Split the data into training and testing sets
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Start MLflow experiment
# mlflow.set_experiment("Walmart_Sales_Prediction")

# with mlflow.start_run():
#     # Log parameters
#     mlflow.log_param("model_type", "LinearRegression")
#     mlflow.log_param("test_size", 0.2)
#     mlflow.log_param("random_state", 42)

#     # Train the model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Calculate metrics
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)

#     # Log metrics
#     mlflow.log_metric("r2_score", r2)
#     mlflow.log_metric("mean_absolute_error", mae)
#     mlflow.log_metric("mean_squared_error", mse)

#     # Log the model
#     mlflow.sklearn.log_model(model, "model")

#     # Log artifacts (e.g., feature importance plot)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=model.coef_, y=features)
#     plt.title("Feature Importance")
#     plt.savefig("feature_importance.png")
#     mlflow.log_artifact("feature_importance.png")

#     print(f"R2 Score: {r2}")
#     print(f"Mean Absolute Error: {mae}")
#     print(f"Mean Squared Error: {mse}")



# import os
# import math
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import dvc.api
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # Set up warnings and plotting
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['figure.figsize'] = [10, 6]

# # Load dataset using DVC
# data_path = 'data/Walmart.csv'
# with dvc.api.open(data_path, mode='r') as f:
#     df = pd.read_csv(f)

# # Display the dataset
# print("Dataset Head:")
# print(df.head())

# # Data preprocessing
# # Fix the date format issue
# df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Use the correct format
# df['weekday'] = df['Date'].dt.weekday
# df['month'] = df['Date'].dt.month
# df['year'] = df['Date'].dt.year
# df.drop(['Date'], axis=1, inplace=True)

# # Define target and features
# target = 'Weekly_Sales'
# features = [i for i in df.columns if i not in [target]]

# # EDA: Target Variable Distribution
# plt.figure(figsize=[8, 4])
# sns.histplot(df[target], color='g', edgecolor="black", linewidth=2, bins=30, kde=True)
# plt.title('Target Variable Distribution - Weekly Sales')
# plt.xlabel('Weekly Sales')
# plt.ylabel('Frequency')
# plt.show()

# # EDA: Visualizing Categorical Features
# print('\033[1mVisualising Categorical Features:\033[0m'.center(100))

# # Identify categorical and numerical features
# nu = df[features].nunique().sort_values()
# cf = [col for col in features if nu[col] <= 45]  # Categorical features
# nf = [col for col in features if nu[col] > 45]   # Numerical features

# n = 2
# plt.figure(figsize=[15, 3 * math.ceil(len(cf) / n)])

# for i in range(len(cf)):
#     plt.subplot(math.ceil(len(cf) / n), n, i + 1)
#     if df[cf[i]].nunique() <= 8:
#         sns.countplot(x=cf[i], data=df, palette='viridis')
#     else:
#         sns.histplot(df[cf[i]], kde=False, bins=20, color='blue')
#     plt.title(f'Distribution of {cf[i]}')
#     plt.xlabel(cf[i])
#     plt.ylabel('Count')

# plt.tight_layout()
# plt.show()

# # EDA: Numeric Features Distribution
# print('\033[1mNumeric Features Distribution\033[0m'.center(130))

# n = 4
# clr = ['r', 'g', 'b', 'g', 'b', 'r']

# plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
# for i in range(len(nf)):
#     plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
#     sns.histplot(df[nf[i]], color=clr[i % len(clr)], edgecolor="black", linewidth=2, bins=10, kde=True)
#     plt.title(f'Distribution of {nf[i]}')
#     plt.xlabel(nf[i])
#     plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# # EDA: Boxplots for Numeric Features
# plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
# for i in range(len(nf)):
#     plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
#     sns.boxplot(y=df[nf[i]], color=clr[i % len(clr)])
#     plt.title(f'Boxplot of {nf[i]}')
#     plt.ylabel(nf[i])

# plt.tight_layout()
# plt.show()

# # EDA: Pairplots for All Features
# g = sns.pairplot(df, diag_kind='kde', corner=True)
# g.map_upper(sns.kdeplot, levels=4, color=".2")
# plt.suptitle('Pairplots for All Features', y=1.02)
# plt.show()

# # Data Preprocessing: Removal of Duplicate Rows
# original_df = df.copy(deep=True)
# rs, cs = original_df.shape

# df.drop_duplicates(inplace=True)

# if df.shape == (rs, cs):
#     print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
# else:
#     print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs - df.shape[0]}')

# # Split the data into training and testing sets
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Start MLflow experiment
# mlflow.set_experiment("Walmart_Sales_Prediction")

# with mlflow.start_run():
#     # Log parameters
#     mlflow.log_param("model_type", "LinearRegression")
#     mlflow.log_param("test_size", 0.2)
#     mlflow.log_param("random_state", 42)

#     # Train the model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Calculate metrics
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)

#     # Log metrics
#     mlflow.log_metric("r2_score", r2)
#     mlflow.log_metric("mean_absolute_error", mae)
#     mlflow.log_metric("mean_squared_error", mse)

#     # Log the model
#     mlflow.sklearn.log_model(model, "model")

#     # Log artifacts (e.g., feature importance plot)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=model.coef_, y=features)
#     plt.title("Feature Importance")
#     plt.savefig("feature_importance.png")
#     mlflow.log_artifact("feature_importance.png")

#     print(f"R2 Score: {r2}")
#     print(f"Mean Absolute Error: {mae}")
#     print(f"Mean Squared Error: {mse}")




import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dvc.api
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFE
import statsmodels.api as sm

# Set up warnings and plotting
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 6]

# Load dataset using DVC
data_path = 'data/Walmart.csv'
with dvc.api.open(data_path, mode='r') as f:
    df = pd.read_csv(f)

# Display the dataset
print("Dataset Head:")
print(df.head())

# Data preprocessing
# Fix the date format issue
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Use the correct format
df['weekday'] = df['Date'].dt.weekday
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df.drop(['Date'], axis=1, inplace=True)

# Define target and features
target = 'Weekly_Sales'
features = [i for i in df.columns if i not in [target]]

# EDA: Target Variable Distribution
plt.figure(figsize=[8, 4])
sns.histplot(df[target], color='g', edgecolor="black", linewidth=2, bins=30, kde=True)
plt.title('Target Variable Distribution - Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.show()

# EDA: Visualizing Categorical Features
print('\033[1mVisualising Categorical Features:\033[0m'.center(100))

# Identify categorical and numerical features
nu = df[features].nunique().sort_values()
cf = [col for col in features if nu[col] <= 45]  # Categorical features
nf = [col for col in features if nu[col] > 45]   # Numerical features

n = 2
plt.figure(figsize=[15, 3 * math.ceil(len(cf) / n)])

for i in range(len(cf)):
    plt.subplot(math.ceil(len(cf) / n), n, i + 1)
    if df[cf[i]].nunique() <= 8:
        sns.countplot(x=cf[i], data=df, palette='viridis')
    else:
        sns.histplot(df[cf[i]], kde=False, bins=20, color='blue')
    plt.title(f'Distribution of {cf[i]}')
    plt.xlabel(cf[i])
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# EDA: Numeric Features Distribution
print('\033[1mNumeric Features Distribution\033[0m'.center(130))

n = 4
clr = ['r', 'g', 'b', 'g', 'b', 'r']

plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
    sns.histplot(df[nf[i]], color=clr[i % len(clr)], edgecolor="black", linewidth=2, bins=10, kde=True)
    plt.title(f'Distribution of {nf[i]}')
    plt.xlabel(nf[i])
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# EDA: Boxplots for Numeric Features
plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
    sns.boxplot(y=df[nf[i]], color=clr[i % len(clr)])
    plt.title(f'Boxplot of {nf[i]}')
    plt.ylabel(nf[i])

plt.tight_layout()
plt.show()

# EDA: Pairplots for All Features
g = sns.pairplot(df, diag_kind='kde', corner=True)
g.map_upper(sns.kdeplot, levels=4, color=".2")
plt.suptitle('Pairplots for All Features', y=1.02)
plt.show()

# Data Preprocessing: Removal of Duplicate Rows
original_df = df.copy(deep=True)
rs, cs = original_df.shape

df.drop_duplicates(inplace=True)

if df.shape == (rs, cs):
    print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
else:
    print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs - df.shape[0]}')

# Check for empty elements
nvc = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values'] / df.shape[0], 3) * 100
print(nvc)

# One-Hot Encoding and Dummy Encoding
df3 = df.copy()

ecc = nvc[nvc['Percentage'] != 0].index.values
fcc = [i for i in cf if i not in ecc]
oh = True
dm = True

for i in fcc:
    if df3[i].nunique() == 2:
        if oh:
            print("\033[1mOne-Hot Encoding on features:\033[0m")
            print(i)
            oh = False
        df3[i] = pd.get_dummies(df3[i], drop_first=True, prefix=str(i))
    if df3[i].nunique() > 2:
        if dm:
            print("\n\033[1mDummy Encoding on features:\033[0m")
            print(i)
            dm = False
        df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df3[i], drop_first=True, prefix=str(i)))], axis=1)

print("\nShape after encoding:", df3.shape)

# Removal of Outliers
df1 = df3.copy()

for i in nf:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3 + (1.5 * IQR))]
    df1 = df1[df1[i] >= (Q1 - (1.5 * IQR))]
    df1 = df1.reset_index(drop=True)

print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(df3.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))

# Final Dataset
df = df1.copy()
df.columns = [i.replace('-', '_') for i in df.columns]

plt.title('Final Dataset')
plt.pie([df.shape[0], original_df.shape[0] - df.shape[0]], radius=1, labels=['Retained', 'Dropped'], counterclock=False,
        autopct='%1.1f%%', pctdistance=0.9, explode=[0, 0], shadow=True)
plt.pie([df.shape[0]], labels=['100%'], labeldistance=-0, radius=0.78)
plt.show()

print(f'\n\033[1mInference:\033[0m After the cleanup process, {original_df.shape[0] - df.shape[0]} samples were dropped, \
while retaining {round(100 - (df.shape[0] * 100 / (original_df.shape[0])), 2)}% of the data.')

# Data Manipulation
df.columns = [i.replace(' ', '_') for i in df.columns]
X = df.drop([target], axis=1)
Y = df[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True, inplace=True)

print('Original set  ---> ', X.shape, Y.shape, '\nTraining set  ---> ', Train_X.shape, Train_Y.shape, '\nTesting set   ---> ', Test_X.shape, '', Test_Y.shape)

# Standardization
std = StandardScaler()

print('\033[1mStandardardization on Training set'.center(120))
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
display(Train_X_std.describe())

print('\n', '\033[1mStandardardization on Testing set'.center(120))
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
display(Test_X_std.describe())

# Feature Selection: Correlation Matrix
print('\033[1mCorrelation Matrix'.center(100))
plt.figure(figsize=[25, 20])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0)
plt.show()

# Feature Selection: Recursive Feature Elimination (RFE)
Train_xy = pd.concat([Train_X_std, Train_Y.reset_index(drop=True)], axis=1)
a = Train_xy.columns.values

API = sm.OLS(Train_Y, sm.add_constant(Train_X_std)).fit()
print(API.summary())

# Polynomial Features and RFE
Trr = []
Tss = []
n = 3
order = ['ord-' + str(i) for i in range(2, n)]
Trd = pd.DataFrame(np.zeros((10, n - 2)), columns=order)
Tsd = pd.DataFrame(np.zeros((10, n - 2)), columns=order)

m = df.shape[1] - 2
for i in range(m):
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=Train_X_std.shape[1] - i)  # Running RFE
    rfe = rfe.fit(Train_X_std, Train_Y)

    LR = LinearRegression()
    LR.fit(Train_X_std.loc[:, rfe.support_], Train_Y)

    pred1 = LR.predict(Train_X_std.loc[:, rfe.support_])
    pred2 = LR.predict(Test_X_std.loc[:, rfe.support_])

    Trr.append(np.sqrt(mean_squared_error(Train_Y, pred1)))
    Tss.append(np.sqrt(mean_squared_error(Test_Y, pred2)))

plt.plot(Trr, label='Train RMSE')
plt.plot(Tss, label='Test RMSE')
plt.legend()
plt.grid()
plt.show()

# Model Training with MLflow
mlflow.set_experiment("Walmart_Sales_Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 100)

    # Train the model
    model = LinearRegression()
    model.fit(Train_X_std, Train_Y)

    # Make predictions
    y_pred = model.predict(Test_X_std)

    # Calculate metrics
    r2 = r2_score(Test_Y, y_pred)
    mae = mean_absolute_error(Test_Y, y_pred)
    mse = mean_squared_error(Test_Y, y_pred)

    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("mean_squared_error", mse)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts (e.g., feature importance plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.coef_, y=X.columns)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print(f"R2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")