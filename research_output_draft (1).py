# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

df = pd.read_csv("Realtime_Drilling_F_15.csv")

df.head()

df.columns

!pip install missingno

import missingno as msno
msno.matrix(df);

df.LITH.value_counts()

plt.style.use('_mpl-gallery')

# Lithologies from top to bottom
lithologies = ['Claystone', 'Marl', 'Limestone', 'Dolomite', 'Sandstone']
counts = [17, 30, 1, 3, 219]
colors = ['mediumslateblue', 'aquamarine', 'red', 'orange', 'blue']

# Reverse everything for correct stacking in 3D (bottom-up plotting)
lithologies_rev = lithologies[::-1]
counts_rev = counts[::-1]
colors_rev = colors[::-1]

# Stack z from bottom (0) upwards based on reversed counts
z = [0] + list(np.cumsum(counts_rev[:-1]))
x = [1] * len(lithologies_rev)
y = [1] * len(lithologies_rev)
dx = np.ones_like(x) * 0.5
dy = np.ones_like(x) * 0.5
dz = counts_rev

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})

# Plot from bottom to top (Sandstone at bottom)
for i in range(len(lithologies_rev)):
    ax.bar3d(x[i], y[i], z[i], dx[i], dy[i], dz[i], color=colors_rev[i])

# Axis and title
ax.set(xticklabels=[], yticklabels=[], zlabel='Depth')
ax.set_title('Lithology Layers (Top to Bottom)', size=18)

# Legend in natural top-to-bottom order (Layer 1 = top)
legend_elements = [
    Patch(facecolor=color, label=f'Layer {i+1} - {lith}')
    for i, (color, lith) in enumerate(zip(colors, lithologies))
]
ax.legend(handles=legend_elements, title='Lithology Layers', fontsize=10, title_fontsize=12, loc='upper left')

plt.tight_layout()
plt.show()

shape = df.shape
info = df.info()
nulls = df.isnull().sum()
head = df.head()

df = df.drop(['Time', 'EditFlag','BIT_DIST','ECDBIT','STRATESUM','LAGMWDIFF','LAGMRDIFF','LAGMTDIFF'], axis=1)

# Confirm remaining columns
print("Remaining columns:\n", df.columns.tolist())

df = pd.get_dummies(df, columns=['LITH'], prefix='LITH')

#  Preview updated dataframe
print(df.head())

df.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Drop non-numeric columns before scaling (e.g., timestamp)
df_cleaned = df.drop(columns=[ 'Depth'], errors='ignore')  # adjust if needed

# Select only numeric columns
numeric_cols = df_cleaned.select_dtypes(include='number').columns

# Apply MinMaxScaler to only numeric columns
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[numeric_cols]), columns=numeric_cols)

# Optionally, reattach Depth or Timestamp if you want them for plotting
df_scaled['Depth'] = df['Depth'].values

corr_matrix = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Drilling Parameters", fontsize=16)
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr, spearmanr, kendalltau

x = df['WOB']
y = df['ROP_AVG']

r_pearson, p_pearson = pearsonr(x, y)

# Spearman correlation
rho_spearman, p_spearman = spearmanr(x, y)

# Kendall correlation
tau_kendall, p_kendall = kendalltau(x, y)

print(f"Pearson's r: {r_pearson:.4f}, p-value: {p_pearson:.4e}")
print(f"Spearman's rho: {rho_spearman:.4f}, p-value: {p_spearman:.4e}")
print(f"Kendall's tau: {tau_kendall:.4f}, p-value: {p_kendall:.4e}")

def correlation_analysis(x, y, name):
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    tau, p_tau = kendalltau(x, y)

    print(f"{name}")
    print(f"  Pearson's r: {r:.4f}, p-value: {p_r:.4e}")
    print(f"  Spearman's rho: {rho:.4f}, p-value: {p_rho:.4e}")
    print(f"  Kendall's tau: {tau:.4f}, p-value: {p_tau:.4e}")
    print("-" * 50)

# List of parameters to test against ROP_AVG
params = ['TORQUE', 'BIT_RPM', 'MOTOR_RPM', 'SURF_RPM']

# Run correlation analysis
for param in params:
    correlation_analysis(df[param], df['ROP_AVG'], param)

print("Missing values in MOTOR_RPM:", df['MOTOR_RPM'].isna().sum())
print("Missing values in MOTOR_RPM:", df['MOTOR_RPM'].isna().sum())
print("Unique values in MOTOR_RPM:", df['MOTOR_RPM'].nunique())
print("\nDescriptive stats for MOTOR_RPM:")
print(df['MOTOR_RPM'].describe())

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print("Constant columns:", constant_cols)

from sklearn.preprocessing import MinMaxScaler

# Select the columns needed for DHS calculation
features = ['ROP_AVG', 'TORQUE', 'WOB', 'TOTGAS', 'FLOWOUT', 'FLOWIN', 'MWIN', 'MWOUT', 'PUMP']

# Apply MinMax scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Calculate DHS
df_scaled['DHS'] = (
    0.2 * df_scaled['ROP_AVG'] +
    0.15 * (1 - df_scaled['TORQUE']) +
    0.15 * df_scaled['WOB'] +
    0.15 * (1 - df_scaled['TOTGAS']) +
    0.15 * df_scaled['FLOWOUT'] / df_scaled['FLOWIN'] +
    0.1 * df_scaled['MWIN'] / df_scaled['MWOUT'] +
    0.1 * df_scaled['PUMP']
)

# Join the DHS values back with original DataFrame if needed
df_with_dhs = df.copy()
df_with_dhs['DHS'] = df_scaled['DHS']

# Display top 10 rows
print(df_with_dhs[['ROP_AVG', 'TORQUE', 'WOB', 'TOTGAS', 'FLOWIN', 'FLOWOUT', 'MWIN', 'MWOUT', 'PUMP', 'DHS']].head(10))

# Fill NaNs with mean of DHS
df_with_dhs['DHS'] = df_with_dhs['DHS'].fillna(df_with_dhs['DHS'].mean())

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_with_dhs['DHS_Cluster'] = kmeans.fit_predict(df_with_dhs[['DHS']])

from sklearn.cluster import KMeans

# Assume df_with_dhs is your dataframe and 'DHS' is the column
kmeans = KMeans(n_clusters=3, random_state=42)
df_with_dhs['DHS_Cluster'] = kmeans.fit_predict(df_with_dhs[['DHS']])

df_with_dhs[['DHS', 'DHS_Cluster']].head()

from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Handle NaN values (if not already done)
df_with_dhs['DHS'] = df_with_dhs['DHS'].fillna(df_with_dhs['DHS'].mean())

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_with_dhs['DHS_Cluster'] = kmeans.fit_predict(df_with_dhs[['DHS']])

# Step 3: Sort cluster centers and map to labels
cluster_centers = kmeans.cluster_centers_.flatten()
sorted_indices = cluster_centers.argsort()

# Create a mapping: lowest center → "Low", middle → "Medium", highest → "High"
cluster_label_map = {sorted_indices[0]: 'Low', sorted_indices[1]: 'Medium', sorted_indices[2]: 'High'}

# Step 4: Map cluster numbers to category labels
df_with_dhs['DHS_Category'] = df_with_dhs['DHS_Cluster'].map(cluster_label_map)

df_with_dhs[['DHS', 'DHS_Cluster', 'DHS_Category']].head()

plt.figure(figsize=(12,6))
# Use the original DataFrame 'df' to access the 'Depth' column
plt.plot(df['Depth'], df_scaled['DHS'], color='green')
plt.xlabel("Depth")
plt.ylabel("Drilling Health Score (0 to 1)")
plt.title("Drilling Health Score vs Depth")
plt.grid(True)
plt.gca().invert_xaxis()  # Optional: to show depth increasing downward
plt.show()

df_scaled['Depth'] = df['Depth'].values  # reattach Depth

plt.figure(figsize=(10,6))
scatter = plt.scatter(df_scaled['Depth'], df_scaled['DHS'], c=df_scaled['DHS'], cmap='RdYlGn')
plt.xlabel("Depth")
plt.ylabel("DHS")
plt.title("DHS Color Coded by Value")
plt.gca().invert_xaxis()
plt.colorbar(scatter, label="DHS")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Define numerical features you want to plot
numerical_features = ['Depth', 'LAGMWT', 'TORQUE', 'MOTOR_RPM', 'WOB', 'ROP_AVG']

# Filter to only those columns that exist in df_cleaned
available_features = [col for col in numerical_features if col in df_cleaned.columns]

# Plot
plt.figure(figsize=(15, 10))
for i, col in enumerate(available_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_cleaned[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

df_scaled['Depth Range'] = pd.cut(df_scaled['Depth'], bins=5)
plt.figure(figsize=(10,5))
sns.boxplot(data=df_scaled, x='Depth Range', y='DHS', palette="viridis")
plt.title("Boxplot of DHS by Depth Range")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.violinplot(y=df_scaled['DHS'], inner="box", color="lightgreen")
plt.title("Violin Plot of DHS")
plt.ylabel("DHS")
plt.grid(True)
plt.show()

df_scaled = df_scaled.dropna(subset=['DHS'])

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'features' is your intended X data
# Create DataFrame from 'features', BUT USE df_scaled
X = pd.DataFrame(df_scaled[features])

# Drop non-numeric columns like intervals or object strings
X = X.select_dtypes(include=[np.number])

# Use the corresponding DHS values from df_scaled for y
y = df_scaled['DHS']  # This ensures consistency in shape

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importances
importance = model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features, palette="viridis")
plt.title("Feature Importance for Predicting DHS")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
import numpy as np

# Run KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['DHS_Cluster'] = kmeans.fit_predict(df_scaled[['DHS']])

# Sort cluster centers to map them properly
centers = kmeans.cluster_centers_.flatten()
sorted_indices = np.argsort(centers)  # index of smallest → largest

# Map actual cluster labels to health levels
label_map = {}
label_map[sorted_indices[0]] = 'Critical'
label_map[sorted_indices[1]] = 'Warning'
label_map[sorted_indices[2]] = 'Safe'

# Assign readable labels
df_scaled['Health_Level'] = df_scaled['DHS_Cluster'].map(label_map)

# Print sample results
print(df_scaled[['DHS', 'DHS_Cluster', 'Health_Level']].head(10))

# Join the DHS values back with original DataFrame if needed
df_with_dhs = df.copy()
df_with_dhs['DHS'] = df_scaled['DHS']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = y.fillna(y.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test data
y_pred = lr.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Optional: view model coefficients
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

# Combine and drop rows where y_train is NaN
train_data = pd.concat([X_train, y_train], axis=1)

# Drop NaNs from the target column
target_column = y_train.name  # get the column name
train_data = train_data.dropna(subset=[target_column])

# Split back into features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

# Confirm no NaNs remain
print("NaNs left in y_train after clean-up:", y_train.isna().sum())

# Get indices where y_train is NOT NaN
valid_index = y_train.dropna().index

# Filter both X_train and y_train
X_train = X_train.loc[valid_index]
y_train = y_train.loc[valid_index]

# Double-check
print("After filtering:")
print("NaNs in y_train:", y_train.isna().sum())
print("Shapes:", X_train.shape, y_train.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

# Linear Regression
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(X, y)

# Random Forest
rf_model = RandomForestRegressor(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=50,
    random_state=42
)
rf_model.fit(X, y)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=5,
    n_estimators=50,
    subsample=1.0,
    random_state=42
)
gb_model.fit(X, y)

# XGBoost
xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.8,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X, y)

models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print(f"--- {name} ---")
    print(f"R² Score      : {r2:.4f}")
    print(f"MAE           : {mae:.4f}")
    print(f"MSE           : {mse:.4f}\n")

from sklearn.model_selection import train_test_split

# Split into training and validation sets
X = df_scaled.select_dtypes(include=[np.number]).drop(columns=['DHS', 'Depth'])
y = df_scaled['DHS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import make_scorer, mean_squared_error

# Use negative MSE as scoring (because GridSearchCV maximizes)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define models and grids
models_grids = {
    "Linear Regression": {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False]
        }
    },
    "Random Forest": {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    "Gradient Boosting": {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    },
    "XGBoost": {
        'model': xgb.XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
}

best_models = {}

for name, entry in models_grids.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(entry['model'], entry['params'], scoring=scorer, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_models[name] = {
        'best_estimator': grid.best_estimator_,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }

    print(f"Best hyperparameters for {name}: {grid.best_params_}\n")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Retrain models with best params
lr_model = LinearRegression(fit_intercept=True)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

gb_model = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)

# Fit all models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}

print("\n📊 Model Performance on Validation Set:\n")

for name, model in models.items():
    y_pred = model.predict(X_val)
    print(f"{name}:")
    print(f"  R² Score : {r2_score(y_val, y_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(y_val, y_pred):.4f}")
    print(f"  MSE      : {mean_squared_error(y_val, y_pred):.4f}")
    print(f"  RMSE     : {np.sqrt(mean_squared_error(y_val, y_pred)):.4f}\n")

from sklearn.ensemble import IsolationForest

# Keep only numeric columns
df_numeric = df_scaled.select_dtypes(include=[np.number])
# Exclude 'Depth' if you want to drop it
df_for_iso = df_numeric.drop(columns=['Depth'])  # or any other column to exclude

iso = IsolationForest(contamination=0.05, random_state=42)
df_scaled['Anomaly'] = iso.fit_predict(df_for_iso)

df_scaled['Anomaly'].value_counts()

df_cleaned['DHS'] = df_with_dhs.loc[df_cleaned.index, 'DHS']

print(df_cleaned[['DHS']].head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# --- Function to drop constant features ---
def drop_constant_columns(df, threshold=0.0):
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    constant_columns = [col for col in df.columns if col not in df.columns[selector.get_support()]]
    df = df.drop(columns=constant_columns)
    return df, constant_columns

# --- Separate features and target ---
X = df_cleaned.drop(columns=['DHS'])
y = df_cleaned['DHS']

# --- Fill NaNs in DHS with the mean ---
y.fillna(y.mean(), inplace=True)
X.fillna(X.mean(), inplace=True)


# --- Separate features and target ---
X = df_cleaned.drop(columns=['DHS'])
y = df_cleaned['DHS']

# --- Drop constant features ---
X, dropped_cols = drop_constant_columns(X)
print("Dropped constant columns:", dropped_cols)

# --- Scale features ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Initialize models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(**{
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }),
    "Gradient Boosting": GradientBoostingRegressor(**{
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'random_state': 42
    }),
    "XGBoost": XGBRegressor(**{
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'random_state': 42
    })
}

# --- Train and evaluate models ---
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred)
    })

# --- Create results DataFrame ---
results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Placeholder for your training and test data (replace with your actual data)
# Example: X_train, X_test = your_features; y_train, y_test = your_target
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.rand(100)      # 100 target values
X_test = np.random.rand(20, 10)    # 20 test samples
y_test = np.random.rand(20)        # 20 test target values

# 1. Linear Regression
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, y_pred_lr)

# 2. Random Forest
rf_model = RandomForestRegressor(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=50,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, y_pred_rf)

# 3. Gradient Boosting
gb_model = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=5,
    n_estimators=50,
    subsample=1.0,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_rmse = mean_squared_error(y_test, y_pred_gb)

# 4. XGBoost
xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.8,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, y_pred_xgb)

# Print performance metrics
print("Model Performance (RMSE):")
print(f"Linear Regression: {lr_rmse:.4f}")
print(f"Random Forest: {rf_rmse:.4f}")
print(f"Gradient Boosting: {gb_rmse:.4f}")
print(f"XGBoost: {xgb_rmse:.4f}")

if 'DHS' in df_scaled.columns:
    df['DHS'] = df_scaled['DHS']
    print("✅ DHS added back to df.")
else:
    print("⚠️ DHS column not found in df_scaled.")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Drop non-numeric and irrelevant columns
df_lr = df.select_dtypes(include=[np.number]).copy()

# Drop rows with missing values in features
df_lr = df_lr.dropna()

# Define features and target
X = df_lr.drop(columns=['DHS'])
y = df_lr['DHS']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# Ensure DHS exists in df_scaled
if 'DHS' not in df_scaled.columns:
    if 'DHS' in df.columns:
        # Align index before assigning
        df_scaled = df_scaled.loc[df.index]  # Match df_scaled to df's index
        df_scaled['DHS'] = df['DHS']
    else:
        print("DHS not found in df.")
else:
    # Fix misalignment before filling
    common_index = df_scaled.index.intersection(df.index)
    missing_dhs = df_scaled.loc[common_index, 'DHS'].isnull()
    df_scaled.loc[common_index[missing_dhs], 'DHS'] = df.loc[common_index[missing_dhs], 'DHS']

# Ensure DHS exists first
if 'DHS' not in df.columns:
    raise ValueError("The target column 'DHS' is missing from the dataset.")

# Separate DHS and numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
if 'DHS' not in numeric_df.columns:
    numeric_df['DHS'] = df['DHS']  # Add DHS back if it was dropped during select_dtypes

# Drop rows with missing DHS
numeric_df = numeric_df.dropna(subset=['DHS'])

# Fill remaining missing values with column means
numeric_df.fillna(numeric_df.mean(), inplace=True)

# Split features and target
X = numeric_df.drop(columns=['DHS'])
y = numeric_df['DHS']

# Normalize features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(f"R² Score      : {r2_score(y_test, y_pred):.4f}")
print(f"MAE           : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE           : {mean_squared_error(y_test, y_pred):.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.scatterplot(data=df_scaled, x='Depth', y='DHS', hue='Anomaly', palette={1:'green', -1:'red'})
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Depth")
plt.ylabel("DHS")
plt.legend(title='Anomaly')
plt.show()

# Simulate ground truth for demo (replace this with real if you have)
df_scaled['True_Label'] = np.where(df_scaled['DHS'] < 0.4, -1, 1)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(df_scaled['True_Label'], df_scaled['Anomaly'])
print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(df_scaled['True_Label'], df_scaled['Anomaly']))

## HEALTH RISK LEVEL CLUSTERING PLOT

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df_scaled['Health_Cluster'] = kmeans.fit_predict(df_scaled[['DHS']])

# Map to readable labels
cluster_map = dict(zip(np.sort(df_scaled.groupby('Health_Cluster')['DHS'].mean().index),
                       ['Critical', 'Warning', 'Safe']))
df_scaled['Health_Level'] = df_scaled['Health_Cluster'].map(cluster_map)

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(x=range(len(df_scaled)), y='DHS', hue='Health_Level', data=df_scaled, palette='coolwarm')
plt.title("DHS over Time (Health Levels)")
plt.xlabel("Sample Index")
plt.ylabel("DHS")
plt.tight_layout()
plt.show()

##TIME SERIES
# This simulates time using index
plt.figure(figsize=(12,6))
sns.lineplot(x=range(len(df_scaled)), y='DHS', data=df_scaled)
plt.title("Time Series of DHS")
plt.xlabel("Sample Index (Simulated Time)")
plt.ylabel("DHS")
plt.tight_layout()
plt.show()

## ACTUAL DHS VS PREDICTED DHS

selected_features = df_scaled.select_dtypes(include='number').columns.drop(['DHS', 'Depth'], errors='ignore')

X = df_scaled[selected_features]
y = df_scaled['DHS']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='red')
plt.title("Actual vs Predicted DHS")
plt.xlabel("Actual DHS")
plt.ylabel("Predicted DHS")
plt.tight_layout()
plt.show()

# Print scores
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Define features and target
selected_features = df_scaled.select_dtypes(include='number').columns.drop(['DHS', 'Depth'], errors='ignore')
X = df_scaled[selected_features]
y = df_scaled['DHS']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get test indices
test_indices = X_test.index

# Create columns and fill with NaN first
enhanced_df['Actual_DHS'] = np.nan
enhanced_df['Predicted_DHS'] = np.nan

# Assign actual and predicted values to test rows
enhanced_df.loc[test_indices, 'Actual_DHS'] = y_test.values
enhanced_df.loc[test_indices, 'Predicted_DHS'] = y_pred

# Fill missing values with the mean of test set predictions and actuals
actual_mean = y_test.mean()
predicted_mean = y_pred.mean()

enhanced_df['Actual_DHS'].fillna(actual_mean, inplace=True)
enhanced_df['Predicted_DHS'].fillna(predicted_mean, inplace=True)

enhanced_df.columns

##FEATURE IMPORTANCE
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
plt.title("Feature Importance for DHS Prediction")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Classify zones
df_scaled['Zone'] = df_scaled['DHS'].apply(lambda x: 'Safe' if x >= 0.65 else ('Warning' if x >= 0.5 else 'Critical'))

# Plot
plt.figure(figsize=(12,6))
sns.scatterplot(x=range(len(df_scaled)), y='DHS', hue='Zone', data=df_scaled,
                palette={'Safe':'green', 'Warning':'orange', 'Critical':'red'})
plt.title("Safe vs Risk Zones")
plt.xlabel("Sample Index")
plt.ylabel("DHS")
plt.tight_layout()
plt.show()

df_scaled['Zone_Group'] = (df_scaled['Zone'] != df_scaled['Zone'].shift()).cumsum()
grouped = df_scaled.groupby(['Zone_Group', 'Zone']).size().reset_index(name='Count')

# Step 2: Get indices of Safe groups with more than 1 sample
safe_groups = grouped[(grouped['Zone'] == 'Safe') & (grouped['Count'] > 1)]['Zone_Group'].tolist()

# Step 3: Mark Safe zones that belong to such groups
df_scaled['Safe_Cluster'] = df_scaled.apply(
    lambda row: 'Clustered Safe' if (row['Zone'] == 'Safe' and row['Zone_Group'] in safe_groups) else row['Zone'],
    axis=1
)

# Step 4: Plot
plt.figure(figsize=(14,6))
sns.scatterplot(x=range(len(df_scaled)), y='DHS', hue='Safe_Cluster', data=df_scaled,
                palette={
                    'Clustered Safe': 'limegreen',
                    'Safe': 'green',
                    'Warning': 'orange',
                    'Critical': 'red'
                })
plt.title("Safe vs Risk Zones (with Clustered Safe Zones Highlighted)")
plt.xlabel("Sample Index")
plt.ylabel("DHS")
plt.legend(title='Zone')
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

# Step 1: Grouping areas
df_scaled['Area_ID'] = (df_scaled['Zone'] != df_scaled['Zone'].shift()).cumsum()

# Step 2: Assign numeric Z-values per area for 3D plot
zone_to_z = {'Critical': 0, 'Warning': 1, 'Safe': 2}
df_scaled['Z'] = df_scaled['Zone'].map(zone_to_z)

# Step 3: 3D Scatter Plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

colors = {'Critical': 'red', 'Warning': 'orange', 'Safe': 'green'}

for zone in df_scaled['Zone'].unique():
    subset = df_scaled[df_scaled['Zone'] == zone]
    ax.scatter(
        subset.index,                      # X-axis: sample index
        subset['DHS'],                     # Y-axis: DHS value
        subset['Z'],                       # Z-axis: zone code
        label=zone,
        color=colors[zone],
        s=40,
        alpha=0.8
    )

# Axis labels and style
ax.set_title("3D Zone Area Visualization")
ax.set_xlabel("Sample Index")
ax.set_ylabel("DHS")
ax.set_zlabel("Zone Area Code")
ax.set_zticks([0, 1, 2])
ax.set_zticklabels(['Critical', 'Warning', 'Safe'])
ax.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Recalculate zone change areas if needed
df_scaled['Area_ID'] = (df_scaled['Zone'] != df_scaled['Zone'].shift()).cumsum()

# Create the figure
plt.figure(figsize=(14, 6))

# Plot DHS line
plt.plot(df_scaled.index, df_scaled['DHS'], label='DHS', color='black', linewidth=1.5)

# Shade background for each Zone area
zone_colors = {'Safe': 'green', 'Warning': 'orange', 'Critical': 'red'}

for area_id, area_data in df_scaled.groupby('Area_ID'):
    start_idx = area_data.index.min()
    end_idx = area_data.index.max()
    zone = area_data['Zone'].iloc[0]
    plt.axvspan(start_idx, end_idx, color=zone_colors[zone], alpha=0.3)

# Labels and formatting
plt.title("DHS Over Time with Zone Areas Highlighted")
plt.xlabel("Sample Index")
plt.ylabel("Drilling Health Score (DHS)")
plt.legend(['DHS'])
plt.tight_layout()
plt.show()

df.to_csv("final_drilling_data.csv", index=False)

df.to_excel("final_drilling_data.xlsx", index=False)

# Make sure DHS_Category and DHS_Cluster are present in df_with_dhs
enhanced_df = df.merge(
    df_with_dhs[['DHS', 'DHS_Category', 'DHS_Cluster']],
    on='DHS',
    how='left'
)

enhanced_df.to_csv("final_drilling_data_3.csv", index=False)

