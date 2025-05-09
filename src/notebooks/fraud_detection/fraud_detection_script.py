import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
from io import StringIO
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Helper Function for Skewness Check & Plotting ---
def check_and_plot_skewness(df, columns, threshold=1.0, plot_dir='plots'):
    """Checks skewness of specified columns and plots distributions for highly skewed ones."""
    print(f"\n--- Checking Skewness (Threshold > {threshold}) ---")
    highly_skewed_cols = []
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory for plots: {plot_dir}")

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found for skewness check.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is not numeric, skipping skewness check.")
            continue

        try:
            skewness = df[col].skew()
            print(f"Skewness of {col}: {skewness:.2f}")

            if abs(skewness) > threshold:
                highly_skewed_cols.append(col)
                print(f"  -> Highly skewed. Plotting original distribution...")
                try:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Original Distribution of {col} (Skewness: {skewness:.2f})')
                    plot_filename = os.path.join(plot_dir, f'original_distribution_{col}.png')
                    plt.savefig(plot_filename)
                    plt.close() # Close the plot to avoid displaying it
                    print(f"     Saved plot to {plot_filename}")
                except Exception as e:
                    print(f"     Warning: Could not plot original distribution for {col}: {e}")
        except Exception as e:
             print(f"Warning: Could not calculate skewness for column '{col}': {e}")

    print("----------------------------------------")
    return highly_skewed_cols

# --- Setup ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# --- Configuration ---
DATASET_PATH = '../../../dataset.csv' # Relative path from script to dataset
TARGET_COLUMN = 'fraud_status_6month' # As defined in yettel_hw.txt
POSITIVE_LABEL = 'Y'
NEGATIVE_LABEL = 'N'
TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.95
N_ITER_RANDOM_SEARCH = 50 # Number of iterations for RandomizedSearchCV
CV_FOLDS = 5

# --- Import Preprocessing Function ---
# Adjust path to import lambda function
try:
    # Assumes the script is run from the workspace root or src/notebooks/fraud_detection/
    # Try relative path from src/notebooks/fraud_detection
    module_path_rel = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lambdas', 'yettel_csv_preprocessor'))
    # Try relative path from workspace root (common in IDEs)
    module_path_root = os.path.abspath(os.path.join('.', 'src', 'lambdas', 'yettel_csv_preprocessor'))

    if os.path.exists(os.path.join(module_path_rel, 'yettel_csv_preprocessor.py')):
        if module_path_rel not in sys.path:
            sys.path.append(module_path_rel)
    elif os.path.exists(os.path.join(module_path_root, 'yettel_csv_preprocessor.py')):
         if module_path_root not in sys.path:
             sys.path.append(module_path_root)
    else:
         raise ImportError("Could not determine path to yettel_csv_preprocessor module.")

    from yettel_csv_preprocessor import preprocess_data
    print("Successfully imported preprocess_data function.")
except ImportError as e:
    print(f"Error importing preprocess_data: {e}")
    print("Attempting to define preprocess_data locally based on previous inspection.")
    # Define preprocess_data locally as a fallback if import fails
    def preprocess_data(df):
        df['day_id'] = pd.to_datetime(df['day_id'])
        categorical_cols = [
            'pl_subseg_desc', 'gender', 'operating_system', 'handset_feature_cat_desc',
            'lmh_desc', 'channel_class', 'channel_group', 'address_county', 'outlet_county'
        ]
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        numeric_cols = [
            'r_age_y', 'instal_cnt', 'moving_average_price_amt',
            'selling_price_amt', 'upfront_pym_amt', 'monthly_fee'
        ]
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        df['price_difference'] = df['moving_average_price_amt'] - df['selling_price_amt']
        # Handle potential division by zero before calculating ratios
        df['payment_ratio'] = np.where(df['selling_price_amt'] != 0, df['upfront_pym_amt'] / df['selling_price_amt'], 0)
        # Handle potential division by zero for monthly burden calculation
        df['monthly_burden'] = np.where(df['instal_cnt'] != 0, (df['selling_price_amt'] - df['upfront_pym_amt']) / df['instal_cnt'] + df['monthly_fee'], df['monthly_fee']) # Assume full burden is monthly fee if instal_cnt is 0
        df['instalment_ind'] = df['instalment_ind'].astype(int)
        return df

# --- 1. Load Data ---
print(f"Loading dataset from: {DATASET_PATH}")
try:
    # Try reading with default separator first
    try:
        df = pd.read_csv(DATASET_PATH, sep=';', decimal='.') # Explicitly set separator and decimal
    except Exception:
        print("Failed reading with ';', trying ',' separator...")
        df = pd.read_csv(DATASET_PATH, sep=',', decimal='.') # Fallback to comma

    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("Dataset head:\\n", df.head())
    print("\\nDataset info:")
    df.info()
    print("\\nInitial missing values:\\n", df.isnull().sum())
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# --- 2. Initial Preprocessing (Lambda Function) ---
print("\\nApplying initial preprocessing...")
df_processed = preprocess_data(df.copy())
print("Initial preprocessing applied.")
print("Missing values after initial preprocessing:\\n", df_processed.isnull().sum())

# --- 3. Target Variable Definition ---
print(f"\\nDefining target variable '{TARGET_COLUMN}'...")
if TARGET_COLUMN in df_processed.columns:
    df_processed['target'] = df_processed[TARGET_COLUMN].apply(lambda x: 1 if x == POSITIVE_LABEL else 0)
    print(f"Target variable '{TARGET_COLUMN}' created and mapped to 'target'.")
    # Calculate and print class imbalance for the whole dataset
    print("\\n--- Overall Dataset Class Imbalance ---")
    overall_counts = df_processed['target'].value_counts()
    overall_percentages = df_processed['target'].value_counts(normalize=True) * 100
    overall_imbalance_summary = pd.DataFrame({'Count': overall_counts, 'Percentage': overall_percentages})
    print(overall_imbalance_summary)
    print("-------------------------------------")

    # --- Plot Overall Class Imbalance --- (Using pre-calculated data)
    try:
        plt.figure(figsize=(6, 4))
        # Use barplot with the summary data
        ax = sns.barplot(x=overall_imbalance_summary.index, y=overall_imbalance_summary['Count'])
        plt.title(f'Overall Class Distribution (Full Dataset)\n0={NEGATIVE_LABEL}, 1={POSITIVE_LABEL}')
        plt.xlabel('Target Class')
        plt.ylabel('Count')
        plt.xticks(ticks=[0, 1], labels=[f'{NEGATIVE_LABEL} (0)', f'{POSITIVE_LABEL} (1)']) # Ensure correct labels

        # Annotate using pre-calculated percentages
        for index, row in overall_imbalance_summary.reset_index().iterrows():
            # ax.text needs x, y, s (string)
            # x-coordinate is the index (0 or 1), y-coordinate is the count (row['Count'])
            ax.text(index, row['Count'] + 3, f"{row['Percentage']:.1f}%", ha='center')

        overall_imbalance_plot_filename = 'overall_class_imbalance.png'
        plt.savefig(overall_imbalance_plot_filename)
        plt.close()
        print(f"Saved overall class imbalance plot to {overall_imbalance_plot_filename}")
    except Exception as e:
        print(f"Warning: Could not plot overall class imbalance: {e}")
    # -------------------------------------

    # print("Target distribution:\\n", df_processed['target'].value_counts(normalize=True)) # Original line, now covered above
else:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    print("Available columns:", df_processed.columns.tolist())
    sys.exit(1)

# --- 4. Further Feature Engineering & Preprocessing ---

# 4.1 Categorical Encoding
print("\\nApplying One-Hot Encoding...")
from sklearn.preprocessing import OneHotEncoder

categorical_cols_to_encode = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
# Exclude IDs, high cardinality text, and original targets/related columns
cols_to_exclude = ['day_id', 'id', 'product_name', 'manufacturer_name_en', TARGET_COLUMN, 'nopay_after_12month']
categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col not in cols_to_exclude]

print(f"Categorical columns to encode: {categorical_cols_to_encode}")

if categorical_cols_to_encode:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df_processed[categorical_cols_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols_to_encode), index=df_processed.index)

    # Drop original categorical columns and concatenate encoded ones
    df_processed = df_processed.drop(columns=categorical_cols_to_encode)
    df_final = pd.concat([df_processed, encoded_df], axis=1)
    print(f"Shape after encoding: {df_final.shape}")
else:
    df_final = df_processed.copy()
    print("No categorical columns found for encoding.")


# 4.2 Handling Potential Infinite Values
print("\\nChecking for infinite values in engineered features...")
cols_to_check_inf = ['payment_ratio', 'monthly_burden']
for col in cols_to_check_inf:
    if col in df_final.columns:
        inf_count = np.isinf(df_final[col]).sum()
        if inf_count > 0:
            print(f"Found {inf_count} infinite values in '{col}'. Replacing with NaN.")
            df_final[col] = df_final[col].replace([np.inf, -np.inf], np.nan)
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Filled NaNs in '{col}' with median value: {median_val}")
        else:
            print(f"No infinite values found in '{col}'.")

# 4.3 Log Transformation (Check Skewness First)
print("\\nChecking skewness and applying log transformation if needed...")

# Identify potential numeric columns (original + engineered)
numeric_cols_original = [
    'r_age_y', 'instal_cnt', 'moving_average_price_amt',
    'selling_price_amt', 'upfront_pym_amt', 'monthly_fee'
]
engineered_features = ['price_difference', 'payment_ratio', 'monthly_burden']
boolean_numeric = ['instalment_ind']
cols_for_log_check = numeric_cols_original + engineered_features
cols_for_log_check = [col for col in cols_for_log_check if col in df_final.columns] # Ensure they exist

# --- Call the skewness check function --- 
skewed_cols_to_transform = check_and_plot_skewness(df_final, cols_for_log_check, threshold=1.0)
# ---------------------------------------

cols_to_scale = cols_for_log_check + boolean_numeric
cols_to_scale = list(set([col for col in cols_to_scale if col in df_final.columns])) # Unique and existing

log_transformed_cols = []

# Apply log transform only to columns identified as skewed by the function
# for col in cols_for_log_check: # Old loop
for col in skewed_cols_to_transform:
    # Make a temporary copy for checking/shifting before log - No longer needed here
    # temp_col_data = df_final[col].copy()
    # Check skewness on potentially shifted data - No longer needed here
    # skewness = temp_col_data.skew()
    # print(f"Skewness of {col} (potentially shifted): {skewness:.2f}")

    # Apply log transform if highly skewed (e.g., abs(skewness) > 1)
    # if abs(skewness) > 1: # Condition already met if col is in skewed_cols_to_transform
    print(f"Applying log1p transformation to original {col} data.")

    # --- Plot original distribution before transformation --- # Plotting is now done in the function
    # try:
    #     plt.figure(figsize=(8, 4))
    #     sns.histplot(df_final[col], kde=True)
    #     plt.title(f'Original Distribution of {col} (Skewness: {df_final[col].skew():.2f})')
    #     plot_filename = f'original_distribution_{col}.png'
    #     plt.savefig(plot_filename)
    #     plt.close() # Close the plot to avoid displaying it if run interactively
    #     print(f"Saved plot of original distribution for {col} to {plot_filename}")
    # except Exception as e:
    #     print(f"Warning: Could not plot original distribution for {col}: {e}")
    # --------------------------------------------------------

    # Apply log1p to the *original* data in df_final before scaling
    # Use log1p to handle original zeros gracefully
    log_col_name = col + '_log'
    df_final[log_col_name] = np.log1p(df_final[col])

    # Check for inf/-inf after log transform (e.g. if original value was negative)
    if np.isinf(df_final[log_col_name]).any():
         print(f"Warning: Inf values detected in '{log_col_name}'. Replacing with NaN and median.")
         df_final[log_col_name] = df_final[log_col_name].replace([np.inf, -np.inf], np.nan)
         df_final[log_col_name] = df_final[log_col_name].fillna(df_final[log_col_name].median())


    # Update scaling list: remove original, add log version
    if col in cols_to_scale:
        cols_to_scale.remove(col)
    cols_to_scale.append(log_col_name)
    log_transformed_cols.append(log_col_name)

    # Drop the original non-log transformed column
    if col in df_final.columns:
         df_final = df_final.drop(columns=[col])

# 4.4 Feature Scaling (Standardization)
print("\\nApplying Standardization...")
cols_to_scale = list(set([col for col in cols_to_scale if col in df_final.columns])) # Ensure unique and exist

if cols_to_scale:
    print(f"Columns to scale: {cols_to_scale}")
    scaler = StandardScaler()
    df_final[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])
    print("Numeric features scaled.")
    # print(df_final[cols_to_scale].describe())
else:
    print("No columns identified for scaling.")


# 4.5 Final Feature Selection & Preparation
print("\\nPreparing final features for modeling...")
# Drop unnecessary columns
cols_to_drop = ['day_id', 'id', 'product_name', 'manufacturer_name_en', TARGET_COLUMN, 'nopay_after_12month']
# Also drop original columns that were log-transformed
original_cols_logged = [c.replace('_log','') for c in log_transformed_cols]
cols_to_drop.extend(original_cols_logged)

df_model = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns], errors='ignore')

# Separate features (X) and target (y)
if 'target' in df_model.columns:
    X = df_model.drop(columns=['target'])
    y = df_model['target']
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Ensure all feature columns are numeric - important for PCA/Model
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in features: {non_numeric_cols}")
        print("Dropping non-numeric columns...")
        X = X.drop(columns=non_numeric_cols)
        print(f"Final features shape after dropping non-numeric: {X.shape}")
    else:
        print("All feature columns are numeric.")
    # print("Final features head:\\n", X.head())

else:
    print("Error: 'target' column not found after preprocessing.")
    sys.exit(1)

# Check for NaN/Inf values before proceeding
if X.isnull().sum().sum() > 0:
    print("\\nWarning: NaN values detected in final features (X). Imputing with median...")
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print("NaN values imputed.")
if np.isinf(X).sum().sum() > 0:
     print("\\nWarning: Infinite values detected in final features (X). Replacing with large finite number...")
     X = X.replace([np.inf, -np.inf], np.finfo(np.float64).max) # Or use median imputation again
     print("Infinite values replaced.")


# --- 5. Train/Validation/Test Split ---
print("\\nSplitting data into training, validation, and testing sets...")
from sklearn.model_selection import train_test_split

# First split into training (80%) and testing (20%)
# We use X and y prepared in step 4.5
X_temp_train, X_test, y_temp_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE, # e.g., 0.2 for 20% test set
    random_state=RANDOM_STATE,
    stratify=y # Stratify based on the target variable y
)

# Calculate validation size relative to the original data (e.g., 16% validation -> 0.2 * 0.8 = 0.16)
# We need the proportion relative to the temp_train set
# If TEST_SIZE is 0.2, temp_train is 0.8. If we want VAL_SIZE_ABS=0.16 (16% of total),
# then VAL_SIZE_REL = 0.16 / 0.8 = 0.2 (20% of the temp_train set)
VAL_SIZE_RELATIVE = TEST_SIZE / (1 - TEST_SIZE) # Make validation set size proportional to test set size within the temp train set

# Split the temporary training set into final training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp_train, y_temp_train,
    test_size=VAL_SIZE_RELATIVE, # e.g., 0.2 of the 80% -> 16% of total
    random_state=RANDOM_STATE,
    stratify=y_temp_train # Stratify based on the temporary train target
)


print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("Train target distribution:\\n", y_train.value_counts(normalize=True))
print("Validation target distribution:\\n", y_val.value_counts(normalize=True))
print("Test target distribution:\\n", y_test.value_counts(normalize=True))

# --- 6. Dimensionality Reduction (PCA) ---
print("\\nApplying PCA...")
from sklearn.decomposition import PCA

# Fit PCA on training data only
n_components_pca = min(X_train.shape[0], X_train.shape[1]) # Max possible components
pca = PCA(n_components=n_components_pca, random_state=RANDOM_STATE)
pca.fit(X_train)

# Determine number of components for desired variance
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_selected = np.argmax(cumsum_variance >= PCA_VARIANCE_THRESHOLD) + 1
print(f"Selected number of PCA components (threshold={PCA_VARIANCE_THRESHOLD:.2f}): {n_components_selected}")

# Apply PCA with selected components
pca_final = PCA(n_components=n_components_selected, random_state=RANDOM_STATE)
X_train_pca = pca_final.fit_transform(X_train)
# Transform validation and test sets using the *same* fitted PCA
X_val_pca = pca_final.transform(X_val)
X_test_pca = pca_final.transform(X_test)

print(f"X_train shape after PCA: {X_train_pca.shape}")
print(f"X_val shape after PCA: {X_val_pca.shape}")
print(f"X_test shape after PCA: {X_test_pca.shape}")

# Use PCA-transformed data for modeling
X_train_final = X_train_pca
X_val_final = X_val_pca # Keep validation set separate
X_test_final = X_test_pca

# --- 7. Model Training (XGBoost with Hyperparameter Tuning) ---
print("\\nTraining XGBoost model with Hyperparameter Tuning...")

# --- Apply SMOTE to Training Data ---
print("\nApplying SMOTE to the training data...")

# --- Visualize Class Imbalance Before SMOTE ---
print("Visualizing class distribution in the training set before SMOTE...")

# Calculate and store training imbalance data first
train_counts = y_train.value_counts()
train_percentages = y_train.value_counts(normalize=True) * 100
train_imbalance_summary = pd.DataFrame({'Count': train_counts, 'Percentage': train_percentages})
print("\nTraining set distribution before SMOTE:")
print(train_imbalance_summary)

# Plot using the pre-calculated data
plt.figure(figsize=(6, 4))
# ax = sns.countplot(x=y_train) # Old method
ax = sns.barplot(x=train_imbalance_summary.index, y=train_imbalance_summary['Count'])
plt.title(f'Class Distribution Before SMOTE (Train Set)\n0={NEGATIVE_LABEL}, 1={POSITIVE_LABEL}')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=[f'{NEGATIVE_LABEL} (0)', f'{POSITIVE_LABEL} (1)']) # Ensure correct labels

# Annotate using pre-calculated percentages
for index, row in train_imbalance_summary.reset_index().iterrows():
    ax.text(index, row['Count'] + 3, f"{row['Percentage']:.1f}%", ha='center')

# Old annotation method
# total = len(y_train)
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width() / 2., height + 3,
#             f'{height*100/total:.1f}%', # Percentage format
#             ha='center')

imbalance_plot_filename = 'class_imbalance_before_smote.png'
plt.savefig(imbalance_plot_filename)
plt.close()
print(f"Saved class imbalance plot to {imbalance_plot_filename}")
# print("Training set distribution before SMOTE:\\n", y_train.value_counts(normalize=True)) # Original line, now covered above
# ---------------------------------------------

smote = SMOTE(random_state=RANDOM_STATE)
# Apply SMOTE only to the training set
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

print(f"Original training shape: {X_train_final.shape}, {y_train.shape}")
print(f"Resampled training shape: {X_train_smote.shape}, {y_train_smote.shape}")
print("Resampled training target distribution:\n", pd.Series(y_train_smote).value_counts(normalize=True))


# --- Define Optuna Objective Function ---
def objective(trial):
    # Define hyperparameters to tune
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'use_label_encoder': False,
        'verbosity': 0, # Suppress XGBoost warnings during tuning
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 regularization
        'random_state': RANDOM_STATE
        # Note: scale_pos_weight is removed as we are using SMOTE
    }

    model = xgb.XGBClassifier(**param)

    # Perform cross-validation on the SMOTE-resampled training data
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Using F1 score for cross-validation scoring as requested
    score = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring='f1', n_jobs=-1)

    return score.mean()

# --- Run Optuna Study ---
print("\nStarting Optuna hyperparameter search...")
# Create study object, maximize F1 score
study = optuna.create_study(direction='maximize', study_name='XGBoost Optimization')
# Increase number of trials for potentially better results
study.optimize(objective, n_trials=N_ITER_RANDOM_SEARCH) # Reuse N_ITER_RANDOM_SEARCH for number of trials

print(f"\nOptuna study finished. Number of finished trials: {len(study.trials)}")

# Get best parameters
best_params = study.best_params
print(f"Best parameters found by Optuna: {best_params}")
print(f"Best F1 score found (CV on SMOTE data): {study.best_value:.4f}")

# Train final model with best parameters on the SMOTE-resampled training data
print("\nTraining final XGBoost model with best parameters found by Optuna...")
best_xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    use_label_encoder=False,
    **best_params, # Unpack best parameters found by Optuna
    random_state=RANDOM_STATE
)
best_xgb_clf.fit(X_train_smote, y_train_smote)
print("Final model training complete.")

# --- 8. Model Evaluation ---
print("\nEvaluating the Optuna fine-tuned model on the original test set...")

# Predict on the test set
# Use X_test_final (PCA transformed test set)
y_pred = best_xgb_clf.predict(X_test_final)
y_pred_proba = best_xgb_clf.predict_proba(X_test_final)[:, 1] # Probabilities for positive class

# --- Classification Report ---
print("\nClassification Report:\n")
# Evaluate against y_test
print(classification_report(y_test, y_pred, target_names=[f'Non-Fraud ({NEGATIVE_LABEL})', f'Fraud ({POSITIVE_LABEL})']))

# --- F1 Score ---
# Evaluate against y_test
f1 = f1_score(y_test, y_pred)
print(f"F1 Score on Test Set: {f1:.4f}")

# --- ROC AUC Score ---
# Evaluate against y_test
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# --- Precision-Recall AUC ---
# Evaluate against y_test
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# --- Confusion Matrix ---
print("\nConfusion Matrix:")
# Evaluate against y_test
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Non-Fraud ({NEGATIVE_LABEL})', f'Fraud ({POSITIVE_LABEL})'],
            yticklabels=[f'Non-Fraud ({NEGATIVE_LABEL})', f'Fraud ({POSITIVE_LABEL})'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# Save or show plot
plt.savefig('confusion_matrix.png')
# plt.show()
print("Confusion matrix saved as confusion_matrix.png")

# --- Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'XGBoost (PR AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
# Save or show plot
plt.savefig('precision_recall_curve.png')
# plt.show()
print("Precision-Recall curve saved as precision_recall_curve.png")


# --- Feature Importances (Optional - from best model) ---
# Note: Feature importances are harder to interpret directly after PCA.
# If you need interpretability, consider running the model on features *before* PCA.
# print("\\nFeature Importances (from best XGBoost model on PCA components):")
# try:
#     importances = best_xgb_clf.feature_importances_
#     feature_names = [f'PC_{i+1}' for i in range(X_train_final.shape[1])]
#     importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#     importance_df = importance_df.sort_values(by='Importance', ascending=False)
#     print(importance_df.head(10))
# except Exception as e:
#     print(f"Could not retrieve feature importances: {e}")


print("\nScript execution finished.") 