import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# -----------------------------
# 1) Page Config + Some Styling
# -----------------------------
st.set_page_config(
    page_title="Pathfinder",
    page_icon=":graduation_cap:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS for modern dark theme and fade-in animation
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 18px;
    font-family: 'Fira Sans', sans-serif;
    background-color: #121212 !important;
    color: #ffffff !important;
}
main > div {
    max-width: 2000px;
    margin: 0 auto;
}
.stSelectbox > div, .stButton > button, .stTextInput > div > div {
    background-color: #222 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #444 !important;
    padding: 8px;
    font-size: 18px;
}
.stButton > button {
    background-color: #7209b7 !important;
    padding: 10px 20px;
    font-size: 18px;
    border-radius: 12px;
    margin-top: 10px;
}
.card-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4rem;
    margin-top: 2rem;
    padding-bottom: 1rem;
}
.card {
    background-color: #213547;
    border-radius: 16px;
    width: 100%;
    max-width: 1800px;
    padding: 2.5rem 4rem;
    border: 1px solid #2c3e50;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    opacity: 0;
    transform: translateY(20px) scale(0.98);
    animation: fadeInUp 0.6s ease-out forwards;
    animation-delay: var(--delay);
}
@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}
.card h4 {
    color: #fff;
    font-size: 30px;
    margin-bottom: 1.25rem;
}
.card p {
    color: #ccc;
    font-size: 20px;
    line-height: 2;
}
.card a {
    color: #9be1ff;
    text-decoration: none;
    font-size: 20px;
}
.card a:hover {
    text-decoration: underline;
}
.loader {
    margin-top: 20px;
    border: 4px solid #2c3e50;
    border-top: 4px solid #00f0ff;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    animation: spin 1s linear infinite;
    margin-left: auto;
    margin-right: auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.status-text {
    text-align: center;
    color: #9be1ff;
    font-size: 20px;
    font-style: italic;
    margin-top: 10px;
    animation: fadein 0.5s ease-in-out;
}
@keyframes fadein {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2) Data Loading and Preprocessing
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("OPPORTUNITIES.xlsx")
    df['tuition'] = pd.to_numeric(df['tuition'], errors='coerce')
    df['tuition'] = df['tuition'].fillna(0)

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

df = load_data()
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])

# -----------------------------
# 3) Header and Inputs
# -----------------------------
st.markdown("<h1 style='text-align: center; font-size: 90px;'>Pathfinder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select your preferences to discover exciting opportunities:</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    user_type = st.selectbox("Type", ['Summer Program', 'Internship'])
    user_field = st.selectbox("Field", ['Stem', 'Humanities', 'Business', 'Arts'])
with col2:
    user_level = st.selectbox("Education Level", ['High School', 'Undergraduate'])
    user_mode = st.selectbox("Participation Mode", ['In-Person', 'Hybrid'])

user_tuition = st.selectbox("Tuition", ['Free', 'Low', 'Medium', 'High', 'Very High'])

# -----------------------------
# 4) Recommendation Results
# -----------------------------
if st.button("Get Recommendations"):
    progress = st.empty()
    loader_placeholder = st.empty()

    progress.markdown("<p class='status-text'>Analyzing preferences...</p>", unsafe_allow_html=True)
    loader_placeholder.markdown("<div class='loader'></div>", unsafe_allow_html=True)
    time.sleep(1.3)

    progress.markdown("<p class='status-text'>Finding matches...</p>", unsafe_allow_html=True)
    time.sleep(1.0)

    progress.markdown("<p class='status-text'>Ranking opportunities...</p>", unsafe_allow_html=True)
    time.sleep(1.0)

    progress.empty()
    loader_placeholder.empty()

    user_query = f"{user_type} {user_level} {user_field} {user_mode} {user_tuition}"
    user_vector = vectorizer.transform([user_query])
    df['similarity_score'] = cosine_similarity(user_vector, feature_matrix)[0]

    results = df.sort_values(by='similarity_score', ascending=False).head(50)

    st.markdown("<div class='card-container'>", unsafe_allow_html=True)
    for i, row in enumerate(results.iterrows()):
        _, r = row
        delay = min(i * 0.07, 0.5)  # Max delay cap 0.5s
        st.markdown(f"""
            <div class="card" style="--delay: {delay}s">
                <h4>{r['program_name']}</h4>
                <p>
                  <strong>Type:</strong> {r['type']}<br>
                  <strong>Education:</strong> {r['education_level']}<br>
                  <strong>Field:</strong> {r['field_of_study']}<br>
                  <strong>Mode:</strong> {r['mode_of_participation']}<br>
                  <strong>Tuition:</strong> {int(r['tuition']) if r['tuition'] else 'Free'}<br>
                  <a href="{r['link']}" target="_blank">🔗 Program Link</a>
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
