import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

if "history" not in st.session_state:
    st.session_state["history"] = []

if "positive_count" not in st.session_state:
    st.session_state["positive_count"] = 0

if "negative_count" not in st.session_state:
    st.session_state["negative_count"] = 0

st.title("🎬 IMDb Film Yorum Analizi")
st.markdown("Film yorumunu gir ve duygu analizini gör!")
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.sidebar.title("Ayarlar")
show_emoji = st.sidebar.checkbox("Emoji ile göster", True)

st.markdown(
    """
    <div style='background-color:#2596be;padding:10px;border-radius:10px'>
        💡 Bu model yorumun pozitif mi negatif mi analiz eder!
    </div>
    """, unsafe_allow_html=True
)

user_input = st.text_area("Yorumunu gir (ENG)")

if st.button("Analyze"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        if sentiment == "Positive":
            st.session_state.positive_count += 1
            st.success(f"Yorumun analizi: {sentiment} ✅")
        else:
            st.session_state.negative_count += 1
            st.error(f"Yorumun analizi: {sentiment} ❌")

        st.session_state.history.append({
            "Yorum": user_input,
            "Sonuç": sentiment
        })

    else:
        st.warning("Please enter a comment")


import pandas as pd
import altair as alt



df = pd.DataFrame({
    "Sentiment": ["Positive", "Negative"],
    "Count": [st.session_state.positive_count, st.session_state.negative_count]
})
st.write(df.head())

# Renk map'i
color_scale = alt.Scale(domain=["Positive","Negative"], range=["green","red"])

# Grafik
chart = alt.Chart(df).mark_bar(size=40).encode(
    x=alt.X("Sentiment:N", title=""),
    y=alt.Y("Count:Q", title="Adet"),
    color=alt.Color("Sentiment:N", scale=color_scale, legend=None)
)

st.altair_chart(chart, use_container_width=True)

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.subheader("📜 Analiz Geçmişi")
    st.write(df_history)