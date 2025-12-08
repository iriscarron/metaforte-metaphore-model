import pandas as pd
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="analyse métaphores",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_data():
    df = pd.read_csv(PROJECT_ROOT / "data/processed/polititweets_top200_metaphors.csv")
    df_full = pd.read_csv(PROJECT_ROOT / "data/processed/polititweets_clean.csv")
    df = df.merge(df_full[['text', 'user_id']], on='text', how='left')
    return df

df = load_data()

st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        font-size: 2.5rem;
        color: #1a1a1a;
        margin-bottom: 2rem;
    }
    .tweet-container {
        border: 1px solid #e0e0e0;
        border-radius: 2px;
        padding: 0;
        margin-bottom: 1rem;
        background-color: #ffffff;
        overflow: hidden;
    }
    .metric-box {
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 2px;
        background-color: #ffffff;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.2rem;
        color: #1a1a1a;
        font-weight: 500;
    }
    .column-header {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .cell-content {
        color: #1a1a1a;
        line-height: 1.5;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("analyse des métaphores")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">tweets analysés</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">score moyen</div>
        <div class="metric-value">{df['metaphor_score'].mean():.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    score_threshold = st.slider("seuil de filtrage", 0.0, 1.0, 0.0, 0.01)

filtered_df = df[df['metaphor_score'] >= score_threshold]

st.markdown("<br><br>", unsafe_allow_html=True)

# affichage des tweets
tweets_par_page = 20
total_pages = (len(filtered_df) - 1) // tweets_par_page + 1 if len(filtered_df) > 0 else 1
page = st.number_input("page", min_value=1, max_value=total_pages, value=1, step=1)

debut = (page - 1) * tweets_par_page
fin = min(debut + tweets_par_page, len(filtered_df))

st.markdown(f"**{len(filtered_df)}** tweets • page {page}/{total_pages}")
st.markdown("<br>", unsafe_allow_html=True)

for idx, row in filtered_df.iloc[debut:fin].iterrows():
    st.markdown(f"""
    <div class="tweet-container">
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="width:50%; padding: 1.25rem; vertical-align:top; border-right: 1px solid #e0e0e0;">
                    <div class="column-header">tweet</div>
                    <div class="cell-content">{row['text']}</div>
                </td>
                <td style="width:17%; padding: 1.25rem; vertical-align:top; border-right: 1px solid #e0e0e0;">
                    <div class="column-header">auteur</div>
                    <div class="cell-content">{row['user_id'] if pd.notna(row['user_id']) else '—'}</div>
                </td>
                <td style="width:17%; padding: 1.25rem; vertical-align:top; border-right: 1px solid #e0e0e0;">
                    <div class="column-header">émotion</div>
                    <div class="cell-content">—</div>
                </td>
                <td style="width:16%; padding: 1.25rem; vertical-align:top;">
                    <div class="column-header">métaphore</div>
                    <div class="cell-content" style="font-weight: 500;">{row['metaphor_score']:.3f}</div>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
