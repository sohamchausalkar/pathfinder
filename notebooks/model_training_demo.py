import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, preprocess_data, split_data

# Load and preprocess data
file_path = 'data/processed_data/opportunities.csv'
df = load_data(file_path)
df = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = split_data(df, 'Opportunity Type')  # 'Opportunity Type' as target column

# Train a decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model if needed
import pickle
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)


