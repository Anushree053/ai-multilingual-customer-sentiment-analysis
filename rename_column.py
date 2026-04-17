import pandas as pd

# Load your dataset
data = pd.read_csv("dataset.csv")

# Rename column to feedback
data = data.rename(columns={"review": "feedback"})
# If your dataset has 'text' column instead of 'review', use:
# data = data.rename(columns={"text": "feedback"})

# Save new dataset
data.to_csv("customer_feedback.csv", index=False)

print("Column renamed successfully")