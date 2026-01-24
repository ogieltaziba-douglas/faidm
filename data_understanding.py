import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset.
print("\n" + "=" * 40)
print("STEP 1: LOADING THE DATASET")
print("=" * 40)
df = pd.read_csv("CDC Diabetes Dataset.csv")

print("\n" + "=" * 40)
print("STEP 2: INITIAL DATA INSPECTION")
print("=" * 40)

print("\nOriginal dataset shape:", df.shape)
print("-" * 40)
print(df.head())

print("\nColumn Names and Data Types:")
print("-" * 40)
print(df.dtypes)

print("\nData Information:")
print("-" * 40)
print(df.info())

# =============================================================================
#  TARGET VARIABLE ANALYSIS
# =============================================================================
print("\n" + "=" * 40)
print("STEP 3: TARGET VARIABLE ANALYSIS")
print("=" * 40)

# The target variable is Diabetes_012
# 0 = No Diabetes, 1 = Pre-Diabetes, 2 = Diabetes
target_col = "Diabetes_012"

print(f"\nTarget Variable: '{target_col}'")
print("-" * 40)
print(f"\nValue Counts:")
target_counts = df[target_col].value_counts().sort_index()
print(target_counts)

print(f"\nPercentage Distribution:")
target_pct = (df[target_col].value_counts(normalize=True) * 100).sort_index()
for val, pct in target_pct.items():
    label = {0.0: "No Diabetes", 1.0: "Pre-Diabetes", 2.0: "Diabetes"}.get(
        val, str(val)
    )
    print(f"  {label} ({int(val)}): {pct:.2f}%")

# Check for class imbalance
print(f"\nCLASS IMBALANCE ANALYSIS:")
imbalance_ratio = target_counts.max() / target_counts.min()
print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
if imbalance_ratio > 3:
    print("SIGNIFICANT IMBALANCE DETECTED - May need to address this during modeling")
else:
    print("Moderate imbalance - Consider stratified sampling")

# =============================================================================
#  STATISTICAL SUMMARY
# =============================================================================
print("\n" + "=" * 40)
print("STEP 4: STATISTICAL SUMMARY")
print("=" * 40)

print("\nDescriptive Statistics:")
print("-" * 40)
desc_stats = df.describe().T
desc_stats["range"] = desc_stats["max"] - desc_stats["min"]
desc_stats["cv"] = (
    desc_stats["std"] / desc_stats["mean"]
) * 100  # Coefficient of variation
print(
    desc_stats[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "range", "cv"]
    ].round(2)
)

# =============================================================================
#  MISSING VALUES ANALYSIS
# =============================================================================
print("\n" + "=" * 40)
print("STEP 5: MISSING VALUES ANALYSIS")
print("=" * 40)

missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({"Missing Count": missing_counts, "Missing %": missing_pct})

total_missing = missing_counts.sum()
if total_missing == 0:
    print("\n NO MISSING VALUES DETECTED - Dataset is complete!")
else:
    print(f"\n MISSING VALUES FOUND:")
    print(missing_df[missing_df["Missing Count"] > 0])

# =============================================================================
#  DUPLICATE ANALYSIS
# =============================================================================
print("\n" + "=" * 40)
print("STEP 6: DUPLICATE ANALYSIS")
print("=" * 40)

duplicates = df.duplicated().sum()
duplicate_pct = (duplicates / len(df)) * 100

print(f"\nDuplicate Rows: {duplicates:,} ({duplicate_pct:.2f}%)")
if duplicates > 0:
    print("RECOMMENDATION: Consider removing duplicates before modeling")
else:
    print("No exact duplicate rows found")


# =============================================================================
#  FEATURE TYPE CLASSIFICATION
# =============================================================================
print("\n" + "=" * 40)
print("STEP 7: FEATURE TYPE CLASSIFICATION")
print("=" * 40)

# Analyze each column to determine if it's binary, categorical, or continuous
feature_types = {}
for col in df.columns:
    unique_vals = df[col].nunique()
    if unique_vals == 2:
        feature_types[col] = "Binary"
    elif unique_vals <= 10:
        feature_types[col] = "Categorical (Ordinal)"
    else:
        feature_types[col] = "Continuous/Numerical"

print("\nFeature Classification:")
print("-" * 40)
for col, ftype in feature_types.items():
    unique = df[col].nunique()
    print(f"  {col}: {ftype} (unique values: {unique})")

# Summary by type
type_counts = pd.Series(feature_types).value_counts()
print(f"\nFeature Type Summary:")
for ftype, count in type_counts.items():
    print(f"  → {ftype}: {count} features")

# =============================================================================
#  OUTLIER DETECTION (for continuous variables)
# =============================================================================
print("\n" + "=" * 40)
print("STEP 8: OUTLIER DETECTION (IQR Method)")
print("=" * 40)

# Focus on continuous variables
continuous_cols = [col for col, ftype in feature_types.items() if "Continuous" in ftype]

print("\nOutlier Analysis for Continuous Variables:")
print("-" * 40)

outlier_info = {}
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100

    outlier_info[col] = {
        "count": outlier_count,
        "percentage": outlier_pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    print(f"  {col}:")
    print(f"    Outliers: {outlier_count:,} ({outlier_pct:.2f}%)")
    print(f"    Valid Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

# =============================================================================
#  CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 40)
print("STEP 9: CORRELATION ANALYSIS")
print("=" * 40)

# Calculate correlation matrix
corr_matrix = df.corr()

# Find features most correlated with target
target_corr = (
    corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
)

print(f"\nTop 10 Features Most Correlated with {target_col}:")
print("-" * 40)
for feature, corr in target_corr.head(10).items():
    direction = "+" if corr > 0 else "-"
    print(f"  {feature}: {corr:.4f} ({direction})")

# Find highly correlated feature pairs (potential multicollinearity)
print(f"\nHighly Correlated Feature Pairs (|r| > 0.5):")
print("-" * 40)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            high_corr_pairs.append(
                {
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": corr_matrix.iloc[i, j],
                }
            )

if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"  {pair['Feature 1']} ↔ {pair['Feature 2']}: {pair['Correlation']:.4f}")
else:
    print("No highly correlated pairs found (|r| > 0.5)")


# =============================================================================
#  VISUALIZATION
# =============================================================================
print("\n" + "=" * 40)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 40)

# Create figure directory
import os

os.makedirs("figures", exist_ok=True)

# Target Variable Distribution
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
target_counts.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
ax.set_title(
    "Distribution of Diabetes Status (Target Variable)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Diabetes Status (0=No, 1=Pre-Diabetes, 2=Diabetes)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
for i, (idx, val) in enumerate(target_counts.items()):
    ax.text(i, val + 1000, f"{val:,}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("figures/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/01_target_distribution.png")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="RdBu_r",
    center=0,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/02_correlation_heatmap.png")

# Feature Distributions by Target Class
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.ravel()

# Select key features to visualize
key_features = [
    "BMI",
    "Age",
    "GenHlth",
    "HighBP",
    "HighChol",
    "PhysHlth",
    "MentHlth",
    "Income",
    "Education",
    "DiffWalk",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
]

for i, feature in enumerate(key_features[:16]):
    for diabetes_val in [0.0, 1.0, 2.0]:
        subset = df[df[target_col] == diabetes_val][feature]
        axes[i].hist(
            subset,
            bins=20,
            alpha=0.5,
            label=f"Diabetes={int(diabetes_val)}",
            edgecolor="black",
        )
    axes[i].set_title(feature, fontsize=10, fontweight="bold")
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=8)

plt.suptitle(
    "Feature Distributions by Diabetes Status", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("figures/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/03_feature_distributions.png")

# Box plots for key numerical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

numerical_features = ["BMI", "Age", "GenHlth", "PhysHlth", "MentHlth", "Income"]
for i, feature in enumerate(numerical_features):
    sns.boxplot(data=df, x=target_col, y=feature, ax=axes[i], palette=colors)
    axes[i].set_title(f"{feature} by Diabetes Status", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Diabetes Status")
    axes[i].set_ylabel(feature)

plt.suptitle(
    "Box Plots: Key Features by Diabetes Status", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("figures/04_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/04_boxplots.png")

# Correlation with Target (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 8))
colors_corr = ["#e74c3c" if x < 0 else "#2ecc71" for x in target_corr]
target_corr.plot(kind="barh", color=colors_corr, edgecolor="black", ax=ax)
ax.set_title(f"Feature Correlation with {target_col}", fontsize=14, fontweight="bold")
ax.set_xlabel("Correlation Coefficient", fontsize=12)
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/05_target_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/05_target_correlation.png")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT: DATA UNDERSTANDING PHASE")
print("=" * 80)

print(
    """
 DATASET OVERVIEW:
   • Name: CDC Diabetes Health Indicators Dataset
   • Shape: {rows:,} rows × {cols} columns
   • Target Variable: Diabetes_012 (0=No, 1=Pre-Diabetes, 2=Diabetes)
   • This is a MULTICLASS CLASSIFICATION problem

 DATA QUALITY:
   • Missing Values: {missing}
   • Duplicates: {duplicates:,} ({dup_pct:.2f}%)
   
 CLASS IMBALANCE:
   • Class 0 (No Diabetes): {c0_pct:.1f}%
   • Class 1 (Pre-Diabetes): {c1_pct:.1f}%
   • Class 2 (Diabetes): {c2_pct:.1f}%
   • RECOMMENDATION: Use stratified sampling and consider class weights

 KEY CORRELATIONS WITH DIABETES:
   • Top positive: GenHlth, HighBP, DiffWalk, BMI, Age
   • These features likely indicate health conditions associated with diabetes

 FEATURE TYPES:
   • Binary Features: {binary_count}
   • Ordinal/Categorical Features: {ordinal_count}
   • Continuous Features: {continuous_count}

 VISUALIZATIONS SAVED TO: ./figures/
""".format(
        rows=len(df),
        cols=len(df.columns),
        missing="None" if total_missing == 0 else f"{total_missing:,} found",
        duplicates=duplicates,
        dup_pct=duplicate_pct,
        c0_pct=target_pct.get(0.0, 0),
        c1_pct=target_pct.get(1.0, 0),
        c2_pct=target_pct.get(2.0, 0),
        binary_count=sum(1 for t in feature_types.values() if t == "Binary"),
        ordinal_count=sum(1 for t in feature_types.values() if "Ordinal" in t),
        continuous_count=sum(1 for t in feature_types.values() if "Continuous" in t),
    )
)
