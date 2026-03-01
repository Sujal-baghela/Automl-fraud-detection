from src.data_loader import DataLoader
from src.cleaner import DataCleaner
from src.eda import AutoEDA
from src.model_selector import AutoModelSelector
from src.fraud_system import AutoMLFraudDetector
from src.shap_explainer import IntelligentSHAP

from sklearn.model_selection import train_test_split

# ===============================
# 1️⃣ Load Data
# ===============================

file_path = "Data/creditcard.csv"

loader = DataLoader(file_path)
cleaner = DataCleaner()
eda = AutoEDA()
model_selector = AutoModelSelector()

df = loader.load_data()

# ===============================
# 2️⃣ Remove Duplicates
# ===============================

df = cleaner.remove_duplicates(df)

# ===============================
# 3️⃣ EDA
# ===============================

numerical_cols, categorical_cols = eda.identify_column_types(df)
eda.plot_distributions(df, numerical_cols)
eda.correlation_heatmap(df, numerical_cols)

# ===============================
# 4️⃣ Define Target
# ===============================

target_column = "Class"

X = df.drop(columns=[target_column])
y = df[target_column]

# ===============================
# 5️⃣ Train / Validation / Test Split
# ===============================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1765,
    random_state=42,
    stratify=y_temp
)

# ===============================
# 6️⃣ Initialize Fraud System
# ===============================

fraud_system = AutoMLFraudDetector(
    model_selector=model_selector,
    objective="cost",
    fraud_loss=10000,
    false_alarm_cost=200,
    manual_threshold=None
)

# ===============================
# 7️⃣ Train System
# ===============================

fraud_system.fit(X_train, y_train, X_val, y_val)

# ===============================
# 8️⃣ Evaluate Model Performance
# ===============================

print("\n==== MODEL PERFORMANCE ====")

test_probs = fraud_system.best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probs >= fraud_system.threshold).astype(int)

print("Selected Threshold:", fraud_system.threshold)
print("Sample Probability:", test_probs[0])
print("Predicted Class:", test_predictions[0])
print("Actual Class:", y_test.iloc[0])

# ===============================
# 9️⃣ SHAP EXPLAINABILITY
# ===============================

print("\n==== SHAP EXPLAINABILITY ====")

# Use the FULL trained pipeline
shap_explainer = IntelligentSHAP(fraud_system.best_model)

# Global explanation (saved as image)
shap_explainer.global_explanation(X_test)

# Local explanation (visual)
print("\nTriggering Local Explanation...")
fraud_indices = y_test[y_test == 1].index

if len(fraud_indices) > 0:
    fraud_index = fraud_indices[0]
    reset_index = list(X_test.index).index(fraud_index)

    print("\n🔎 Explaining First Fraud Case...")
    shap_explainer.local_explanation(
        X_test.reset_index(drop=True),
        index=reset_index
    )
else:
    print("No fraud cases found in test set.")

# ===============================
# 🔟 Structured JSON Explanation
# ===============================

json_explanation = shap_explainer.local_explanation_json(
    X_test.reset_index(drop=True),
    index=0
)

# Add prediction info to JSON
json_explanation["prediction_probability"] = float(test_probs[0])
json_explanation["threshold"] = float(fraud_system.threshold)
json_explanation["fraud_prediction"] = bool(test_predictions[0])

print("\n==== STRUCTURED EXPLANATION OUTPUT ====\n")
print(json_explanation)
