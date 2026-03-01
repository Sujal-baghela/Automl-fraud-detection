import joblib
import pandas as pd

# Load trained model
model = joblib.load("best_model.pkl")

print("Model loaded successfully.")

# Example new data (change values as needed)
new_data = pd.DataFrame({
    "Age": [30],
    "Income": [50000],
    "LoanAmount": [200000],
    "CreditScore": [700],
    "EmploymentStatus": ["Employed"],   # must match EXACT spelling
    "MaritalStatus": ["Single"]     
    # You must match columns used during training
})

prediction = model.predict(new_data)

print("Prediction:", prediction[0])
