"""
================================================================================
PHASE 2: DATA PREPARATION (CRISP-DM Phase 3)
================================================================================
This script performs data cleaning and transformation to prepare the dataset
for machine learning modeling.

Key Steps:
1. Remove duplicate rows
2. Convert to binary classification (Diabetes vs No Diabetes)
3. Apply StandardScaler for feature normalization
4. Perform stratified train/test split
5. Save prepared datasets for modeling phases

RATIONALE: All transformations are justified based on Phase 1 EDA findings.
================================================================================
"""

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving scaler and prepared data
import os

print("=" * 80)
print("PHASE 2: DATA PREPARATION")
print("=" * 80)

# =============================================================================
# STEP 2: LOAD THE DATASET
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2: LOADING THE RAW DATASET")
print("=" * 50)

df = pd.read_csv("CDC Diabetes Dataset.csv")
print(f"\nOriginal Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# STEP 3: REMOVE DUPLICATES
# =============================================================================
print("\n" + "=" * 50)
print("STEP 3: REMOVING DUPLICATE ROWS")
print("=" * 50)

# Count duplicates before removal
duplicates_before = df.duplicated().sum()
print(
    f"\nDuplicate rows found: {duplicates_before:,} ({duplicates_before / len(df) * 100:.2f}%)"
)

# Remove duplicates
df_clean = df.drop_duplicates().copy()

# Verify removal
duplicates_after = df_clean.duplicated().sum()
print(f"Duplicate rows after removal: {duplicates_after}")
print(f"Rows removed: {len(df) - len(df_clean):,}")
print(
    f"\nDataset Shape After Deduplication: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns"
)

"""
RATIONALE FOR DUPLICATE REMOVAL:
--------------------------------
1. Duplicates may represent data collection artifacts or survey errors
2. Keeping duplicates could lead to data leakage during cross-validation
3. Duplicates inflate certain patterns, potentially biasing the model
4. Removal ensures each observation is unique and independent
"""

# =============================================================================
# STEP 4: CONVERT TO BINARY CLASSIFICATION
# =============================================================================
print("\n" + "=" * 50)
print("STEP 4: CONVERTING TO BINARY CLASSIFICATION")
print("=" * 50)

print("\nOriginal Class Distribution:")
print("-" * 40)
original_dist = df_clean["Diabetes_012"].value_counts().sort_index()
for val, count in original_dist.items():
    pct = count / len(df_clean) * 100
    label = {0.0: "No Diabetes", 1.0: "Pre-Diabetes", 2.0: "Diabetes"}.get(
        val, str(val)
    )
    print(f"  Class {int(val)} ({label}): {count:,} ({pct:.2f}%)")

# Convert: 0 = No Diabetes, 1 = Diabetes (combining pre-diabetes and diabetes)
# This is stored in a new column 'Diabetes_Binary'
df_clean["Diabetes_Binary"] = df_clean["Diabetes_012"].apply(
    lambda x: 0 if x == 0 else 1
)

print("\nBinary Class Distribution:")
print("-" * 40)
binary_dist = df_clean["Diabetes_Binary"].value_counts().sort_index()
for val, count in binary_dist.items():
    pct = count / len(df_clean) * 100
    label = "No Diabetes" if val == 0 else "Diabetes/Pre-Diabetes"
    print(f"  Class {val} ({label}): {count:,} ({pct:.2f}%)")

# Calculate new imbalance ratio
imbalance_ratio = binary_dist.max() / binary_dist.min()
print(f"\nNew Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"Improvement: From 46:1 (3-class) to {imbalance_ratio:.2f}:1 (binary)")

"""
RATIONALE FOR BINARY CONVERSION:
--------------------------------
1. Pre-Diabetes class (1.8%) is severely underrepresented
2. 3-class classification would be unreliable for Class 1 predictions
3. Binary classification (Diabetes vs No Diabetes) is more practical
4. Clinically, both pre-diabetes and diabetes require intervention
5. Increases minority class from 13.9% to ~15.8%, improving balance
"""

# =============================================================================
# STEP 5: SEPARATE FEATURES AND TARGET
# =============================================================================
print("\n" + "=" * 50)
print("STEP 5: SEPARATING FEATURES AND TARGET")
print("=" * 50)

# Define feature columns (all except the target columns)
feature_cols = [
    col for col in df_clean.columns if col not in ["Diabetes_012", "Diabetes_Binary"]
]

# Create feature matrix (X) and target vector (y)
X = df_clean[feature_cols]
y = df_clean["Diabetes_Binary"]

print(f"\nFeatures (X): {X.shape[0]:,} samples × {X.shape[1]} features")
print(f"Target (y): {y.shape[0]:,} samples")
print(f"\nFeature Columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2}. {col}")

# =============================================================================
# STEP 6: TRAIN/TEST SPLIT WITH STRATIFICATION
# =============================================================================
print("\n" + "=" * 50)
print("STEP 6: TRAIN/TEST SPLIT (80/20 with Stratification)")
print("=" * 50)

# Perform stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,  # For reproducibility
    stratify=y,  # Maintains class proportions in both sets
)

print(
    f"\nTraining Set: {X_train.shape[0]:,} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
)
print(f"Test Set: {X_test.shape[0]:,} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

# Verify stratification
print("\nClass Distribution Verification:")
print("-" * 40)
train_dist = y_train.value_counts(normalize=True) * 100
test_dist = y_test.value_counts(normalize=True) * 100

print(f"Training Set - Class 0: {train_dist[0]:.2f}%, Class 1: {train_dist[1]:.2f}%")
print(f"Test Set     - Class 0: {test_dist[0]:.2f}%, Class 1: {test_dist[1]:.2f}%")
print(f"\nStratification successful - Class proportions maintained!")

"""
RATIONALE FOR STRATIFIED SPLIT:
-------------------------------
1. Random split could result in unequal class proportions
2. Stratification ensures both training and test sets reflect original distribution
3. Prevents test set from being dominated by majority class
4. Essential for imbalanced datasets like ours
5. 80/20 split provides sufficient data for both training and evaluation
"""

# =============================================================================
# STEP 7: FEATURE SCALING (StandardScaler)
# =============================================================================
print("\n" + "=" * 50)
print("STEP 7: FEATURE SCALING (StandardScaler)")
print("=" * 50)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit on training data ONLY, then transform both sets
# This prevents data leakage from test set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for easier inspection
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, columns=feature_cols, index=X_train.index
)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print("\nScaling Statistics (Training Set):")
print("-" * 40)
print(f"Mean of scaled features: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"Std of scaled features: {X_train_scaled.std():.6f} (should be ~1)")

print("\nBefore vs After Scaling (Sample Features):")
print("-" * 40)
sample_features = ["BMI", "Age", "MentHlth", "PhysHlth", "Income"]
print("\n  Feature       | Original Range        | Scaled Range")
print("  " + "-" * 55)
for feat in sample_features:
    orig_min = X_train[feat].min()
    orig_max = X_train[feat].max()
    scaled_min = X_train_scaled_df[feat].min()
    scaled_max = X_train_scaled_df[feat].max()
    print(
        f"  {feat:12} | [{orig_min:6.1f}, {orig_max:6.1f}]     | [{scaled_min:6.2f}, {scaled_max:6.2f}]"
    )

"""
RATIONALE FOR STANDARDSCALER:
-----------------------------
1. Features have different scales:
   - BMI: 12-98
   - Age: 1-13 (categorical)
   - Binary features: 0-1
   - MentHlth/PhysHlth: 0-30

2. Why StandardScaler over MinMaxScaler:
   - Our data has outliers (BMI, MentHlth, PhysHlth)
   - StandardScaler is robust to outliers (uses mean/std)
   - MinMaxScaler would compress normal data due to outliers

3. Why fit on training data only:
   - Prevents data leakage from test set
   - Simulates real-world scenario where test data is unseen
   - Ensures fair model evaluation

4. Which algorithms require scaling:
   - K-Means (distance-based)
   - SVM (kernel computations)
   - KNN (distance-based)
   - Note: Random Forest does NOT require scaling (tree-based)
"""

# =============================================================================
# STEP 8: SAVE PREPARED DATA
# =============================================================================
print("\n" + "=" * 50)
print("STEP 8: SAVING PREPARED DATA")
print("=" * 50)

# Create directory for prepared data
os.makedirs("prepared_data", exist_ok=True)

# Save scaled data as CSV for transparency
X_train_scaled_df.to_csv("prepared_data/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("prepared_data/X_test_scaled.csv", index=False)
y_train.to_csv("prepared_data/y_train.csv", index=False)
y_test.to_csv("prepared_data/y_test.csv", index=False)

# Save unscaled data (for Random Forest which doesn't need scaling)
X_train.to_csv("prepared_data/X_train_unscaled.csv", index=False)
X_test.to_csv("prepared_data/X_test_unscaled.csv", index=False)

# Save the scaler for future use (deployment scenario)
joblib.dump(scaler, "prepared_data/standard_scaler.pkl")

# Save feature names for reference
pd.DataFrame({"feature": feature_cols}).to_csv(
    "prepared_data/feature_names.csv", index=False
)

print("\n Files saved to ./prepared_data/:")
print("-" * 40)
for f in os.listdir("prepared_data"):
    size = os.path.getsize(f"prepared_data/{f}")
    print(f"  • {f} ({size / 1024:.1f} KB)")

# =============================================================================
# STEP 9: SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2 SUMMARY: DATA PREPARATION COMPLETE")
print("=" * 80)

print(f"""
DATA TRANSFORMATION SUMMARY:
   
   Original Dataset:     {df.shape[0]:,} rows × {df.shape[1]} columns
   After Deduplication:  {df_clean.shape[0]:,} rows (removed {len(df) - len(df_clean):,} duplicates)
   
   Target Conversion:    3-class → Binary (No Diabetes vs Diabetes/Pre-Diabetes)
   New Class Balance:    Class 0: {binary_dist[0]:,} ({binary_dist[0] / len(df_clean) * 100:.1f}%)
                         Class 1: {binary_dist[1]:,} ({binary_dist[1] / len(df_clean) * 100:.1f}%)
   
   Train/Test Split:     80% / 20% (Stratified)
   Training Samples:     {X_train.shape[0]:,}
   Test Samples:         {X_test.shape[0]:,}
   
   Scaling Method:       StandardScaler (z-score normalization)
   
OUTPUT FILES:
   • prepared_data/X_train_scaled.csv
   • prepared_data/X_test_scaled.csv
   • prepared_data/y_train.csv
   • prepared_data/y_test.csv
   • prepared_data/standard_scaler.pkl


""")
