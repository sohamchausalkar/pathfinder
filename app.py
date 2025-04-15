import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    file_path = 'OPPORTUNITIES.xlsx'  
    df = pd.read_excel(file_path)

    df['tuition'] = pd.to_numeric(df['tuition'], errors='coerce')
    df['tuition'].fillna(0, inplace=True)

    bins = [-0.1, 0.1, 1000, 4000, 8000, float('inf')]
    labels = ['Free', 'Low', 'Medium', 'High', 'Very High']
    df['tuition_category'] = pd.cut(df['tuition'], bins=bins, labels=labels)


    for col in ['type', 'education_level', 'field_of_study', 'mode_of_participation']:
        df[col] = df[col].astype(str).str.strip().str.title()


    allowed = {
        'type': ['Summer Program', 'Internship'],
        'education_level': ['High School', 'Undergraduate'],
        'field_of_study': ['Stem', 'Humanities', 'Business', 'Arts'],
        'mode_of_participation': ['In-Person', 'Hybrid']
    }
    for col, valid in allowed.items():
        df = df[df[col].isin(valid)]

    df['combined_features'] = (
        df['type'] + ' ' +
        df['education_level'] + ' ' +
        df['field_of_study'] + ' ' +
        df['mode_of_participation'] + ' ' +
        df['tuition_category'].astype(str)
    )

    return df

# ------------------------------
# Main App Logic
# ------------------------------
st.title("🎓 Pathfinder Recommender")
st.markdown("Select your preferences below to discover the most relevant opportunities for you.")

df = load_data()
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])

# User inputs
col1, col2 = st.columns(2)
with col1:
    user_type = st.selectbox("Program Type", ['Summer Program', 'Internship'])
    user_field = st.selectbox("Field of Study", ['Stem', 'Humanities', 'Business', 'Arts'])
with col2:
    user_level = st.selectbox("Education Level", ['High School', 'Undergraduate'])
    user_mode = st.selectbox("Mode of Participation", ['In-Person', 'Hybrid'])

user_tuition = st.selectbox("Tuition Category", ['Free', 'Low', 'Medium', 'High', 'Very High'])

if st.button("🔍 Show Recommendations"):
    user_query = f"{user_type} {user_level} {user_field} {user_mode} {user_tuition}"
    user_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, feature_matrix)
    df['similarity_score'] = similarity_scores[0]

    recommendations = df.sort_values(by='similarity_score', ascending=False).head(10)
    st.success("Top 10 opportunities matching your preferences:")
    st.dataframe(recommendations[['program_name', 'type', 'education_level', 'field_of_study', 'mode_of_participation', 'tuition', 'similarity_score']])
