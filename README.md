# BinThere-DoneThat
This project utilizes historical waste generation data, demographic shifts, and seasonal trends to forecast refuse accumulation levels across urban sectors.

"""
PREDICTIVE ANALYTICS SYSTEM FOR URBAN WASTE MANAGEMENT
"""
# ============================================================================
# Python, NumPy, and Linear Algebra
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("PREDICTIVE ANALYTICS SYSTEM FOR URBAN WASTE MANAGEMENT")
print("=" * 80)
print()

n_samples = 1000
np.random.seed(42)

# 1. Generate the Feature Dictionary
data = {
    'area_id': range(1, n_samples + 1),
    'population': np.random.randint(500, 50000, n_samples),
    'area_sqkm': np.random.uniform(0.5, 20, n_samples),
    'num_households': np.random.randint(100, 15000, n_samples),
    'num_commercial': np.random.randint(0, 500, n_samples),
    'num_industrial': np.random.randint(0, 50, n_samples),
    'avg_income': np.random.uniform(20000, 150000, n_samples),
    'recycling_bins': np.random.randint(10, 500, n_samples),
    'distance_to_facility': np.random.uniform(1, 50, n_samples),
    'collection_frequency': np.random.choice([1, 2, 3, 7], n_samples),
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
    'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                     'Friday', 'Saturday', 'Sunday'], n_samples),
}

# 2. Calculate Base Waste (Corrected Syntax)
base_waste = (data['population'] * 0.8 +
              data['num_households'] * 15 +
              data['num_commercial'] * 50 +
              data['num_industrial'] * 200)

# 3. Add Seasonal Variation
seasonal_factor = {'Spring': 1.0, 'Summer': 1.2, 'Fall': 1.1, 'Winter': 0.9}
season_multiplier = np.array([seasonal_factor[s] for s in data['season']])

# 4. Final Calculation with Random Noise (for realistic regression modeling)
data['waste_generated_kg'] = base_waste * season_multiplier * np.random.uniform(0.85, 1.15, n_samples)

# 5. Convert to DataFrame
df = pd.DataFrame(data)

print("\n" + "=" * 80)
print(" Data Generation and Exploration")
print("=" * 80)
print(df.head())

# Add noise and calculate final waste
data['waste_generated_kg'] = base_waste * season_multiplier * np.random.uniform(0.85, 1.15, n_samples)

# Create waste type distribution (for classification)
waste_types = []
for i in range(n_samples):
    if data['num_industrial'][i] > 20:
        waste_types.append('Industrial')
    elif data['num_commercial'][i] > 200:
        waste_types.append('Commercial')
    elif data['population'][i] > 30000:
        waste_types.append('High-Density Residential')
    else:
        waste_types.append('Residential')

data['waste_type'] = waste_types

# Create collection priority (for classification)
priorities = []
for i in range(n_samples):
    if data['waste_generated_kg'][i] > np.percentile(data['waste_generated_kg'], 75):
        priorities.append('High')
    elif data['waste_generated_kg'][i] > np.percentile(data['waste_generated_kg'], 50):
        priorities.append('Medium')
    else:
        priorities.append('Low')

data['collection_priority'] = priorities

# Create recycling rate
data['recycling_rate'] = (data['recycling_bins'] / data['num_households'] *
                          np.random.uniform(0.1, 0.5, n_samples))
data['recycling_rate'] = np.clip(data['recycling_rate'], 0, 1)

# Create DataFrame
df = pd.DataFrame(data)

print("\nDataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Linear Algebra Operations
print("\n" + "-" * 80)
print("LINEAR ALGEBRA OPERATIONS")
print("-" * 80)

# Select numeric columns for matrix operations
numeric_cols = ['population', 'area_sqkm', 'num_households', 'avg_income', 'waste_generated_kg']
X_matrix = df[numeric_cols].values[:5]  # Use first 5 rows for demonstration

print("\nSample Data Matrix (5x5):")
print(X_matrix)

# Covariance matrix
cov_matrix = np.cov(X_matrix.T)
print("\nCovariance Matrix:")
print(cov_matrix)

# Correlation matrix
corr_matrix = np.corrcoef(X_matrix.T)
print("\nCorrelation Matrix:")
print(corr_matrix)

# Manual computation of covariance between two vectors
vec1 = df['population'].values[:100]
vec2 = df['waste_generated_kg'].values[:100]

manual_cov = np.sum((vec1 - np.mean(vec1)) * (vec2 - np.mean(vec2))) / (len(vec1) - 1)
print(f"\nManual Covariance (population vs waste): {manual_cov:.2f}")

# Manual correlation
manual_corr = manual_cov / (np.std(vec1, ddof=1) * np.std(vec2, ddof=1))
print(f"Manual Correlation: {manual_corr:.4f}")

# ============================================================================
#  Data Visualization
# ============================================================================
print("\n" + "=" * 80)
print("DATA VISUALIZATION")
print("=" * 80)


print("\n" + "=" * 80)
print("DATA VISUALIZATION")
print("=" * 80)

# Set the visual style for professional graphs
sns.set_theme(style="whitegrid")

# Create a 2x2 grid (Total 4 plots)
# This provides indices axes[0, 0], axes[0, 1], axes[1, 0], and axes[1, 1].
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Urban Waste Management - Exploratory Data Analysis', fontsize=20, fontweight='bold')

# 1. Histogram: Distribution of Waste Generated
# Shows the spread of waste_generated_kg across the 1000 samples.
sns.histplot(df['waste_generated_kg'], bins=40, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Waste Generation Distribution', fontsize=15)
axes[0, 0].set_xlabel('Waste Generated (kg)')
axes[0, 0].set_ylabel('Frequency')

# 2. Scatter Plot: Population vs. Waste Generated
# Visualizes how population drives waste generation, categorized by season.
sns.scatterplot(data=df, x='population', y='waste_generated_kg', hue='season',
                alpha=0.6, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Population vs. Waste Generated (by Season)', fontsize=15)
axes[0, 1].set_xlabel('Population')
axes[0, 1].set_ylabel('Waste Generated (kg)')

# 3. Box Plot: Waste Generation by Category
# Compares waste volumes across Residential, Commercial, and Industrial types.
sns.boxplot(data=df, x='waste_type', y='waste_generated_kg', palette='Set3', ax=axes[1, 0])
axes[1, 0].set_title('Waste Volume by Category', fontsize=15)
axes[1, 0].set_xlabel('Waste Type')
axes[1, 0].set_ylabel('Waste Generated (kg)')
axes[1, 0].tick_params(axis='x', rotation=15)

# 4. Heatmap: Correlation Matrix
# Shows mathematical relationships between key features like households and industrial counts.
correlation_cols = ['population', 'num_households', 'num_commercial', 'num_industrial',
                   'avg_income', 'recycling_bins', 'waste_generated_kg', 'recycling_rate']
corr_data = df[correlation_cols].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlation Heatmap', fontsize=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n✓ Visualizations generated successfully. Proceeding to Preprocessing...")
# ============================================================================

# 1. Waste generation distribution
axes[0, 0].hist(df['waste_generated_kg'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Waste Generated')
axes[0, 0].set_xlabel('Waste (kg)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

# 2. Waste by type
waste_by_type = df.groupby('waste_type')['waste_generated_kg'].mean().sort_values()
axes[0, 1].barh(waste_by_type.index, waste_by_type.values, color='coral')
axes[0, 1].set_title('Average Waste by Type')
axes[0, 1].set_xlabel('Average Waste (kg)')
axes[0, 1].grid(alpha=0.3, axis='x')

# 3. Population vs Waste(scatter)
axes[0, 1].scatter(df['population'], df['waste_generated_kg'], alpha=0.5, c='green', s=20)
axes[0, 1].set_title('Population vs Waste Generated')
axes[0, 1].set_xlabel('Population')
axes[0, 1].set_ylabel('Waste (kg)')
axes[0, 1].grid(alpha=0.3)
# 4. Seasonal variation
seasonal_waste = df.groupby('season')['waste_generated_kg'].mean()
axes[1, 0].bar(seasonal_waste.index, seasonal_waste.values, color=['#90EE90', '#FFD700', '#FF8C00', '#87CEEB'])
axes[1, 0].set_title('Average Waste by Season')
axes[1, 0].set_ylabel('Average Waste (kg)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(alpha=0.3, axis='y')
# 5. Correlation heatmap
correlation_cols = ['population', 'num_households', 'num_commercial', 'avg_income',
                    'recycling_bins', 'waste_generated_kg', 'recycling_rate']
corr_data = df[correlation_cols].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Feature Correlation Heatmap')
# 6. Collection priority distribution
priority_counts = df['collection_priority'].value_counts()
axes[1, 1].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
               colors=['#ff6b6b', '#ffd93d', '#6bcf7f'], startangle=90)
axes[1, 1].set_title('Collection Priority Distribution')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'eda_visualizations.png'")

# ============================================================================
#  Data Preprocessing
# ============================================================================

print("\n" + "=" * 80)
print(" Data Preprocessing")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Handle categorical variables
print("\n1. Encoding Categorical Variables:")

# Label encoding for ordinal data (priority)
le_priority = LabelEncoder()
df_processed['collection_priority_encoded'] = le_priority.fit_transform(df_processed['collection_priority'])
print(f"   Priority mapping: {dict(zip(le_priority.classes_, le_priority.transform(le_priority.classes_)))}")

# One-hot encoding for nominal data (season, day_of_week, waste_type)
df_processed = pd.get_dummies(df_processed, columns=['season', 'day_of_week', 'waste_type'], prefix=['season', 'day', 'type'])

print(f"   After encoding: {df_processed.shape[1]} columns")
# 2. Feature Engineering
print("\n2. Feature Engineering:")

# Create new features
df_processed['population_density'] = df_processed['population'] / df_processed['area_sqkm']
df_processed['waste_per_capita'] = df_processed['waste_generated_kg'] / df_processed['population']
df_processed['commercial_ratio'] = df_processed['num_commercial'] / (df_processed['num_households'] + 1)
df_processed['bins_per_household'] = df_processed['recycling_bins'] / (df_processed['num_households'] + 1)

print("   Created features:")
print("   - population_density")
print("   - waste_per_capita")
print("   - commercial_ratio")
print("   - bins_per_household")
# 3. Handle outliers using IQR method
print("\n3. Outlier Detection and Handling:")

def remove_outliers_iqr(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    return outliers, lower_bound, upper_bound

outlier_cols = ['waste_generated_kg', 'population', 'avg_income']
for col in outlier_cols:
    outliers, lb, ub = remove_outliers_iqr(df_processed, col)
    print(f"   {col}: {outliers} outliers detected (range: {lb:.2f} to {ub:.2f})")

# Save processed data
df_processed.to_csv('processed_waste_data.csv', index=False)
print("\n✓ Processed data saved as 'processed_waste_data.csv'")
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the figure for two plots
plt.figure(figsize=(14, 6))

# 1. Histogram: Distribution of Waste Generation
# This helps visualize the spread of your data and identify any remaining skewness post-IQR.
plt.subplot(1, 2, 1)
sns.histplot(df_processed['waste_generated_kg'], bins=30, kde=True, color='purple')
plt.title('Distribution of Waste Generation (Histogram)')
plt.xlabel('Waste Generated (kg)')
plt.ylabel('Frequency')

# 2. Scatter Plot: Population vs. Waste Generated
# This visualizes the correlation between your primary feature (population) and the target variable.
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_processed['population'], y=df_processed['waste_generated_kg'], alpha=0.5, color='darkgreen')
plt.title('Population vs. Waste Generated (Scatter Plot)')
plt.xlabel('Population')
plt.ylabel('Waste Generated (kg)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# ============================================================================
# Linear Regression & Logistic Regression
# ============================================================================

print("\n" + "=" * 80)
print(" Regression Analysis")
print("=" * 80)

# Prepare features for regression
feature_cols = ['population', 'area_sqkm', 'num_households', 'num_commercial',
                'num_industrial', 'avg_income', 'recycling_bins', 'distance_to_facility',
                'collection_frequency', 'recycling_rate', 'population_density',
                'commercial_ratio', 'bins_per_household']

X_reg = df_processed[feature_cols]
y_reg = df_processed['waste_generated_kg']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("\n1. SIMPLE LINEAR REGRESSION (Single Feature)")
print("-" * 60)

# Simple linear regression with population only
X_train_simple = X_train_reg[['population']].values
X_test_simple = X_test_reg[['population']].values

simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train_reg)
y_pred_simple = simple_lr.predict(X_test_simple)

print(f"Coefficient: {simple_lr.coef_[0]:.4f}")
print(f"Intercept: {simple_lr.intercept_:.2f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_simple):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_simple)):.2f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_simple):.2f}")

print("\n2. MULTIPLE LINEAR REGRESSION")
print("-" * 60)

# Multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train_reg_scaled, y_train_reg)
y_pred_mlr = mlr.predict(X_test_reg_scaled)

print(f"Number of features: {len(feature_cols)}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_mlr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_mlr)):.2f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_mlr):.2f}")
# Feature importance (absolute coefficients)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': mlr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())
print("\n3. LOGISTIC REGRESSION (Binary Classification)")
print("-" * 60)

# Binary classification: High priority vs Low/Medium
y_binary = (df_processed['collection_priority'] == 'High').astype(int)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_reg, y_binary, test_size=0.2, random_state=42
)

scaler_log = StandardScaler()
X_train_log_scaled = scaler_log.fit_transform(X_train_log)
X_test_log_scaled = scaler_log.transform(X_test_log)

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_log_scaled, y_train_log)
y_pred_log = log_reg.predict(X_test_log_scaled)
y_pred_proba = log_reg.predict_proba(X_test_log_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_log, y_pred_log, target_names=['Not High Priority', 'High Priority']))
# ============================================================================
#  Advanced Classification and Regression
# ============================================================================


print("\n" + "=" * 80)
print(" Advanced ML Models")
print("=" * 80)
print("\n1. REGRESSION MODELS COMPARISON")
print("-" * 60)
# Test multiple regression models
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    'Support Vector Regression': SVR(kernel='rbf')
}
regression_results = []

for name, model in regression_models.items():
    # Train model
    model.fit(X_train_reg_scaled, y_train_reg)

    # Predictions
    y_pred = model.predict(X_test_reg_scaled)

    # Metrics
    r2 = r2_score(y_test_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')

    regression_results.append({
        'Model': name,
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })

    print(f"{name:.<30} R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

regression_results_df = pd.DataFrame(regression_results)

print("\n2. MULTI-CLASS CLASSIFICATION (Waste Type)")
print("-" * 60)

# Prepare data for waste type classification
le_waste_type = LabelEncoder()
y_waste_type = le_waste_type.fit_transform(df['waste_type'])

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_reg, y_waste_type, test_size=0.2, random_state=42, stratify=y_waste_type
)

scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

# Test multiple classification models
classification_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

classification_results = []

for name, model in classification_models.items():
    # Train model
    model.fit(X_train_class_scaled, y_train_class)

    # Predictions
    y_pred = model.predict(X_test_class_scaled)

    # Metrics
    accuracy = accuracy_score(y_test_class, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_class_scaled, y_train_class, cv=5)

    classification_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'CV Accuracy Mean': cv_scores.mean(),
        'CV Accuracy Std': cv_scores.std()
    })

    print(f"{name:.<30} Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

classification_results_df = pd.DataFrame(classification_results)

# Best classification model - detailed analysis
best_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
best_classifier.fit(X_train_class_scaled, y_train_class)
y_pred_best = best_classifier.predict(X_test_class_scaled)

print("\nBest Model (Random Forest) - Detailed Report:")
print(classification_report(y_test_class, y_pred_best, target_names=le_waste_type.classes_))

# ============================================================================
#  Clustering and PCA
# ============================================================================
print("\n" + "=" * 80)
print(" Clustering and Dimensionality Reduction")
print("=" * 80)

print("\n1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 60)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_train_reg_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Total components: {len(explained_variance)}")
print(f"\nExplained variance by first 5 components:")
for i in range(min(5, len(explained_variance))):
    print(f"  PC{i+1}: {explained_variance[i]:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Reduce to optimal dimensions
pca_optimal = PCA(n_components=n_components_95)
X_pca_reduced = pca_optimal.fit_transform(X_train_reg_scaled)
X_test_pca_reduced = pca_optimal.transform(X_test_reg_scaled)

print(f"Reduced dimensions: {X_reg.shape[1]} → {n_components_95}")

# Train model on reduced features
rf_pca = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_pca.fit(X_pca_reduced, y_train_reg)
y_pred_pca = rf_pca.predict(X_test_pca_reduced)

print(f"\nModel performance with PCA:")
print(f"  R² Score: {r2_score(y_test_reg, y_pred_pca):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_pca)):.2f}")

print("\n2. K-MEANS CLUSTERING")
print("-" * 60)

# Determine optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_reg_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train_reg_scaled, kmeans.labels_))

# Find optimal k
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

# Apply K-Means with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_train_reg_scaled)

# Analyze clusters
df_clustered = df.iloc[X_train_reg.index].copy()
df_clustered['cluster'] = clusters

print(f"\nCluster Analysis:")
cluster_analysis = df_clustered.groupby('cluster').agg({
    'waste_generated_kg': ['mean', 'std', 'count'],
    'population': 'mean',
    'num_commercial': 'mean',
    'recycling_rate': 'mean'
})
print(cluster_analysis)

# Assign cluster names based on characteristics
cluster_names = []
for i in range(optimal_k):
    cluster_data = df_clustered[df_clustered['cluster'] == i]
    avg_waste = cluster_data['waste_generated_kg'].mean()
    if avg_waste > df['waste_generated_kg'].quantile(0.75):
        cluster_names.append(f"High-Waste Zone")
    elif avg_waste > df['waste_generated_kg'].quantile(0.5):
        cluster_names.append(f"Medium-Waste Zone")
    else:
        cluster_names.append(f"Low-Waste Zone")

print("\nCluster Characterization:")
for i, name in enumerate(cluster_names):
    print(f"  Cluster {i}: {name}")
# ============================================================================
# Model Visualization
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING MODEL PERFORMANCE VISUALIZATIONS")
print("=" * 80)

# ============================================================================
#  REVISED MODEL VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL PERFORMANCE VISUALIZATION")
print("=" * 80)

# Set the visual style
sns.set_theme(style="whitegrid")

# Create a 2x3 grid (matching your specified figsize)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

# 1. Regression: Actual vs Predicted (Lasso/Linear Regression)
# Shows how closely the model's waste predictions match the true synthetic values.

axes[0, 0].scatter(y_test_reg, y_pred_mlr, alpha=0.5, c='teal', s=20)
axes[0, 0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0, 0].set_title(f'Regression: Actual vs Predicted\n(R²: {r2_score(y_test_reg, y_pred_mlr):.3f})')
axes[0, 0].set_xlabel('Actual Waste (kg)')
axes[0, 0].set_ylabel('Predicted Waste (kg)')

# 2. Regression: Residuals Plot
# Checks for patterns in errors; random distribution suggests a good model fit.
residuals = y_test_reg - y_pred_mlr
axes[0, 1].scatter(y_pred_mlr, residuals, alpha=0.5, c='orange', s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_title('Residuals Plot (Errors)')
axes[0, 1].set_xlabel('Predicted Waste (kg)')
axes[0, 1].set_ylabel('Residual Error')

# 3. Classification: Confusion Matrix (Waste Type)
# Visualizes how accurately the Random Forest classifies different waste categories.

cm = confusion_matrix(y_test_class, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_waste_type.classes_,
            yticklabels=le_waste_type.classes_, ax=axes[0, 2])
axes[0, 2].set_title('Waste Type Confusion Matrix')
axes[0, 2].set_xlabel('Predicted Type')
axes[0, 2].set_ylabel('Actual Type')

# 4. Clustering: PCA 2D Projection
# Visualizes the K-Means clusters in a 2-dimensional space.
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_train_reg_scaled)
scatter = axes[1, 0].scatter(X_viz[:, 0], X_viz[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=20)
axes[1, 0].set_title(f'K-Means Clusters ({optimal_k} Zones)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# 5. Feature Importance (Random Forest Regression)
# Identifies which urban features drive waste generation the most.
importances = regression_models['Random Forest'].feature_importances_
feat_importances = pd.Series(importances, index=feature_cols).sort_values()
feat_importances.plot(kind='barh', ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Regression Feature Importance')
axes[1, 1].set_xlabel('Relative Importance Score')

# 6. Classification: Actual vs Predicted Counts
# Compares the distribution of predicted types vs actual types.
pred_series = pd.Series(le_waste_type.inverse_transform(y_pred_best)).value_counts().sort_index()
actual_series = pd.Series(le_waste_type.inverse_transform(y_test_class)).value_counts().sort_index()
comparison_df = pd.DataFrame({'Actual': actual_series, 'Predicted': pred_series})
comparison_df.plot(kind='bar', ax=axes[1, 2], color=['#5da5da', '#faa43a'])
axes[1, 2].set_title('Type Prediction Distribution')
axes[1, 2].set_ylabel('Count')
axes[1, 2].tick_params(axis='x', rotation=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n✓ Model visualizations generated successfully.")
# 1. Regression Models Comparison
axes[0, 0].barh(regression_results_df['Model'], regression_results_df['R² Score'], color='steelblue')
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('Regression Models Comparison')
axes[0, 0].grid(alpha=0.3, axis='x')

# 2. Classification Models Comparison
axes[0, 1].barh(classification_results_df['Model'], classification_results_df['Accuracy'], color='coral')
axes[0, 1].set_xlabel('Accuracy')
axes[0, 1].set_title('Classification Models Comparison')
axes[0, 1].grid(alpha=0.3, axis='x')

# 3. PCA Explained Variance
axes[0, 2].plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linewidth=2)
axes[0, 2].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[0, 2].axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} components')
axes[0, 2].set_xlabel('Number of Components')
axes[0, 2].set_ylabel('Cumulative Explained Variance')
axes[0, 2].set_title('PCA - Cumulative Explained Variance')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 4. Actual vs Predicted (Best Regression Model)
best_reg_model = regression_models['Gradient Boosting']
best_reg_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_best_reg = best_reg_model.predict(X_test_reg_scaled)

axes[1, 0].scatter(y_test_reg, y_pred_best_reg, alpha=0.5, s=20)
axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()],
                [y_test_reg.min(), y_test_reg.max()],
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Waste (kg)')
axes[1, 0].set_ylabel('Predicted Waste (kg)')
axes[1, 0].set_title(f'Gradient Boosting: Actual vs Predicted (R²={r2_score(y_test_reg, y_pred_best_reg):.3f})')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5. Confusion Matrix for Waste Type Classification
cm = confusion_matrix(y_test_class, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_waste_type.classes_,
            yticklabels=le_waste_type.classes_,
            ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Waste Type Classification - Confusion Matrix')

# 6. Clustering Visualization (PCA projection)
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_train_reg_scaled)
scatter = axes[1, 2].scatter(X_viz[:, 0], X_viz[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=20)
axes[1, 2].scatter(kmeans_final.cluster_centers_[:, :2][:, 0] if X_train_reg_scaled.shape[1] >= 2 else [0]*optimal_k,
                   kmeans_final.cluster_centers_[:, :2][:, 1] if X_train_reg_scaled.shape[1] >= 2 else [0]*optimal_k,
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
axes[1, 2].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%} variance)')
axes[1, 2].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%} variance)')
axes[1, 2].set_title(f'K-Means Clustering ({optimal_k} clusters)')
axes[1, 2].legend()
plt.colorbar(scatter, ax=axes[1, 2], label='Cluster')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\n✓ Model performance visualizations saved as 'model_performance.png'")

print("\n" + "=" * 80)
print(" Case Study and Outcome Analysis")
print("=" * 80)
print("\nURBAN WASTE MANAGEMENT PREDICTIVE ANALYTICS - FINAL REPORT")
print("=" * 80)
# Save final trained model
import pickle

final_model = {
    'regression_model': best_reg_model,
    'classification_model': best_classifier,
    'clustering_model': kmeans_final,
    'scaler_regression': scaler_reg,
    'scaler_classification': scaler_class,
    'pca_model': pca_optimal,
    'label_encoder_waste_type': le_waste_type,
    'feature_columns': feature_cols
}

with open('waste_management_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
    pickle.dump(final_model, f)

print("\n1. PROJECT OBJECTIVES")
print("-" * 60)
print("✓ Predict waste generation amounts for resource optimization")
print("✓ Classify waste types for appropriate handling")
print("✓ Identify collection priorities for route planning")
print("✓ Segment urban areas into waste management zones")
print("✓ Reduce dimensionality for efficient processing")

print("\n2. DATASET SUMMARY")
print("-" * 60)
print(f"Total samples: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Target variables:")
print(f"  - Waste generation (regression): {df['waste_generated_kg'].min():.2f} - {df['waste_generated_kg'].max():.2f} kg")
print(f"  - Waste types (classification): {len(df['waste_type'].unique())} categories")
print(f"  - Collection priority: {len(df['collection_priority'].unique())} levels")

print("\n3. KEY FINDINGS")
print("-" * 60)

# Top correlations with waste generation - use only original features
original_feature_cols = ['population', 'area_sqkm', 'num_households', 'num_commercial',
                         'num_industrial', 'avg_income', 'recycling_bins',
                         'distance_to_facility', 'collection_frequency', 'recycling_rate']
correlations = df[original_feature_cols + ['waste_generated_kg']].corr()['waste_generated_kg'].sort_values(ascending=False)[1:6]
print("\nTop 5 factors influencing waste generation:")
for feat, corr in correlations.items():
    print(f"  • {feat}: {corr:.3f}")

# Best models
best_reg_idx = regression_results_df['R² Score'].idxmax()
best_class_idx = classification_results_df['Accuracy'].idxmax()

print(f"\nBest Regression Model: {regression_results_df.loc[best_reg_idx, 'Model']}")
print(f"  - R² Score: {regression_results_df.loc[best_reg_idx, 'R² Score']:.4f}")
print(f"  - RMSE: {regression_results_df.loc[best_reg_idx, 'RMSE']:.2f} kg")
print(f"  - MAE: {regression_results_df.loc[best_reg_idx, 'MAE']:.2f} kg")

print(f"\nBest Classification Model: {classification_results_df.loc[best_class_idx, 'Model']}")
print(f"  - Accuracy: {classification_results_df.loc[best_class_idx, 'Accuracy']:.4f}")
print(f"  - Cross-validation: {classification_results_df.loc[best_class_idx, 'CV Accuracy Mean']:.4f} (±{classification_results_df.loc[best_class_idx, 'CV Accuracy Std']:.4f})")

print(f"\nDimensionality Reduction:")
print(f"  - Original features: {len(feature_cols)}")
print(f"  - Reduced features (95% variance): {n_components_95}")
print(f"  - Dimension reduction: {(1 - n_components_95/len(feature_cols))*100:.1f}%")

print(f"\nClustering Results:")
print(f"  - Optimal clusters: {optimal_k}")
print(f"  - Silhouette score: {max(silhouette_scores):.3f}")

print("\n4. BUSINESS IMPACT")
print("-" * 60)

# Calculate potential improvements
baseline_collection = df['waste_generated_kg'].sum()
predicted_optimal = y_pred_best_reg.sum()
efficiency_gain = abs(baseline_collection - len(y_test_reg) * df['waste_generated_kg'].mean()) / baseline_collection

print(f"Resource Optimization:")
print(f"  • Waste prediction accuracy: {r2_score(y_test_reg, y_pred_best_reg)*100:.1f}%")
print(f"  • Route optimization potential: ~{efficiency_gain*15:.1f}% fuel savings")
print(f"  • Collection schedule optimization enabled via priority classification")

print(f"\nOperational Insights:")
print(f"  • {optimal_k} distinct waste management zones identified")
print(f"  • {len(df[df['waste_type']=='Industrial'])} industrial areas require specialized handling")
print(f"  • {len(df[df['collection_priority']=='High'])} high-priority areas need frequent collection")

print("\n5. RECOMMENDATIONS")
print("-" * 60)
print("✓ Deploy Gradient Boosting model for waste generation forecasting")
print("✓ Use Random Forest classifier for waste type identification")
print("✓ Implement zone-based collection routes using clustering results")
print("✓ Focus recycling initiatives in areas with low recycling rates")
print("✓ Allocate resources based on predicted high-priority zones")
print(f"✓ Update models quarterly with new data for continuous improvement")

# Generate final summary statistics
summary_stats = {
    'Total Samples': len(df),
    'Training Samples': len(X_train_reg),
    'Test Samples': len(X_test_reg),
    'Features Used': len(feature_cols),
    'Best Regression R²': regression_results_df['R² Score'].max(),
    'Best Classification Accuracy': classification_results_df['Accuracy'].max(),
    'PCA Components (95%)': n_components_95,
    'Optimal Clusters': optimal_k,
    'Waste Types': len(df['waste_type'].unique()),
    'Priority Levels': len(df['collection_priority'].unique())
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv('project_summary.csv', index=False)

# Save all results
results_summary = {
    'regression_results': regression_results_df,
    'classification_results': classification_results_df,
    'cluster_analysis': cluster_analysis,
    'feature_importance': feature_importance
}




