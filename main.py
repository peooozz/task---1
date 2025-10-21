"""
TASK 1: PREDICT RESTAURANT RATINGS
Machine Learning Internship - Cognifyz Technologies

Objective: Build a machine learning model to predict aggregate rating
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TASK 1: RESTAURANT RATING PREDICTION")
print("="*70)

# ============================================================================
# STEP 1: LOAD THE DATASET
# ============================================================================
print("\n[STEP 1] Loading Dataset...")

df = pd.read_csv('Dataset.csv')
print(f"✓ Dataset loaded successfully")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Display first few rows
print("\nFirst 3 rows:")
print(df.head(3))

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("[STEP 2] Data Preprocessing")
print("="*70)

# 2.1 Check missing values
print("\nChecking missing values...")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print("Missing values found:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("✓ No missing values found!")

# 2.2 Handle missing values in Cuisines (important feature)
if df['Cuisines'].isnull().sum() > 0:
    print(f"\nFilling missing Cuisines with 'Unknown'...")
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# 2.3 Select features for prediction
print("\n" + "-"*70)
print("Selecting Features for Model...")

# Numerical features
numerical_features = ['Average Cost for two', 'Price range', 'Votes']

# Categorical features (we'll encode these)
categorical_features = ['Country Code', 'Has Table booking', 'Has Online delivery']

# Target variable
target = 'Aggregate rating'

print(f"\nNumerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}")
print(f"Target Variable: {target}")

# 2.4 Create feature matrix X and target y
print("\nPreparing feature matrix...")

# Handle missing values in numerical columns
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} missing values with median: {median_val}")

# Encode categorical variables (convert Yes/No to 1/0)
for col in ['Has Table booking', 'Has Online delivery']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    
# Handle any remaining NaN in categorical columns
df[categorical_features] = df[categorical_features].fillna(0)

# Combine all features
X = df[numerical_features + categorical_features].copy()
y = df[target].copy()

print(f"\n✓ Feature matrix created: X shape = {X.shape}")
print(f"✓ Target variable: y shape = {y.shape}")

# Display feature statistics
print("\nFeature Statistics:")
print(X.describe())

# ============================================================================
# STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n" + "="*70)
print("[STEP 3] Splitting Data (80% train, 20% test)")
print("="*70)

# Manual train-test split (80-20)
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(len(X))
split_idx = int(0.8 * len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 4: BUILD AND TRAIN THE MODEL
# ============================================================================
print("\n" + "="*70)
print("[STEP 4] Training Linear Regression Model")
print("="*70)

# Simple Linear Regression implementation
class SimpleLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train the model using Normal Equation"""
        # Add bias term
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Normal Equation: w = (X^T X)^(-1) X^T y
        self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        print("✓ Model trained successfully!")
    
    def predict(self, X):
        """Make predictions"""
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias @ self.weights

# Train the model
model = SimpleLinearRegression()
model.fit(X_train.values, y_train.values)

# ============================================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("[STEP 5] Making Predictions on Test Set")
print("="*70)

y_pred = model.predict(X_test.values)

# Show some predictions vs actual
print("\nSample Predictions vs Actual Ratings:")
print("-" * 50)
print(f"{'Actual':<10} {'Predicted':<12} {'Difference':<12}")
print("-" * 50)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    diff = abs(actual - predicted)
    print(f"{actual:<10.2f} {predicted:<12.2f} {diff:<12.2f}")

# ============================================================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("[STEP 6] Model Evaluation")
print("="*70)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared (R²)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, mae, r2

mse, rmse, mae, r2 = calculate_metrics(y_test.values, y_pred)

print("\nPerformance Metrics:")
print("-" * 50)
print(f"Mean Squared Error (MSE):       {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.4f}")
print(f"R-squared (R²):                 {r2:.4f}")
print("-" * 50)

# Interpretation
print("\nMetrics Interpretation:")
print(f"• The model's average prediction error is {mae:.2f} rating points")
print(f"• R² score of {r2:.4f} means the model explains {r2*100:.2f}% of variance")

if r2 > 0.5:
    print("• Model performance: GOOD ✓")
elif r2 > 0.3:
    print("• Model performance: MODERATE ⚠")
else:
    print("• Model performance: NEEDS IMPROVEMENT ✗")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("[STEP 7] Feature Importance Analysis")
print("="*70)

# Get feature weights (exclude bias)
feature_weights = model.weights[1:]
feature_names = numerical_features + categorical_features

# Sort by absolute importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': feature_weights,
    'Absolute Weight': np.abs(feature_weights)
}).sort_values('Absolute Weight', ascending=False)

print("\nMost Influential Features:")
print("-" * 50)
for idx, row in importance_df.iterrows():
    feature = row['Feature']
    weight = row['Weight']
    direction = "increases" if weight > 0 else "decreases"
    print(f"{feature:<25} Weight: {weight:>8.4f} ({direction} rating)")

# ============================================================================
# STEP 8: SUMMARY AND INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("TASK 1 COMPLETED SUCCESSFULLY! 🎉")
print("="*70)

print("\nKey Findings:")
print(f"1. Dataset contains {len(df)} restaurants")
print(f"2. Model trained on {len(X_train)} samples")
print(f"3. Model tested on {len(X_test)} samples")
print(f"4. Average prediction error: {mae:.2f} rating points")
print(f"5. Model explains {r2*100:.1f}% of rating variance")
print(f"\n6. Most important features:")
for idx, row in importance_df.head(3).iterrows():
    print(f"   - {row['Feature']}")

print("\nNext Steps:")
print("• Try different algorithms (Decision Tree, Random Forest)")
print("• Add more features (Cuisines, Location)")
print("• Tune hyperparameters for better performance")
print("\n" + "="*70)