import pandas as pd
import pickle

def load_model(model_path):
    """Loads the pre-trained decision tree model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_user_input():
    """Collects user input for recommendation."""
    field_of_study = input("Enter your field of study: ")
    skills = input("Enter your key skills (comma separated): ")
    location = input("Preferred location: ")
    return {
        'Field of Study': field_of_study,
        'Skills': skills.split(","),
        'Location': location
    }

def preprocess_user_input(user_input, df_columns):
    """Preprocesses user input to match model's expected input format."""
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    
    # Ensure the input matches the training columns
    missing_cols = set(df_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    input_df = input_df[df_columns]  # Reorder to match the model's column order
    
    return input_df

# Load the model
model = load_model('models/decision_tree_model.pkl')

# Load data to get column structure
df = pd.read_csv('data/processed_data/opportunities.csv')
df = pd.get_dummies(df)
df_columns = df.drop(columns=['Opportunity Type']).columns

# Get user input
user_input = get_user_input()

# Preprocess user input
processed_input = preprocess_user_input(user_input, df_columns)

# Make a prediction
opportunity_type = model.predict(processed_input)[0]
print(f"Recommended Opportunity Type: {opportunity_type}")
