import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon=":material/school:",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    .main-title {
        font-family: 'Cairo', sans-serif;
        font-size: 40px;
        font-weight: 700;
        color: #273338;
        text-align: center;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        color: #618764;
        font-size: 15px;
        letter-spacing: 0.3px;
        margin-bottom: 28px;
    }

    .section-label {
        font-family: 'Cairo', sans-serif;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #273338;
        border-bottom: 2px solid #9CB080;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    .result-card {
        border-radius: 6px;
        padding: 32px;
        text-align: center;
        border: 1px solid;
    }

    .result-grade {
        font-family: 'Cairo', sans-serif;
        font-size: 56px;
        font-weight: 700;
        line-height: 1;
        margin: 8px 0;
    }

    .result-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #618764;
    }

    .result-confidence {
        font-size: 15px;
        color: #273338;
        margin-top: 10px;
    }

    .footer-text {
        text-align: center;
        color: #618764;
        font-size: 12px;
        letter-spacing: 0.3px;
        margin-top: 8px;
    }

    div[data-testid="stSidebar"] {
        background-color: #2B5748;
    }

    div[data-testid="stSidebar"] * {
        color: #F2F4F1 !important;
        font-family: 'Cairo', sans-serif;
    }

    div[data-testid="stSidebar"] hr {
        border-color: #618764;
    }

    .stButton button {
        background-color: #9CB080;
        color: #273338;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        letter-spacing: 0.5px;
        font-family: 'Cairo', sans-serif;
    }

    .stButton button:hover {
        background-color: #618764;
        color: #F2F4F1;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODEL_PATH = "model.pkl"

GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

GRADE_COLORS = {
    "A": "#2B5748",
    "B": "#618764",
    "C": "#9CB080",
    "D": "#B08F4A",
    "F": "#A1503F",
}

GENDER_OPTIONS = ["Male", "Female"]
ETHNICITY_OPTIONS = ["Caucasian", "African American", "Asian", "Other"]
PARENT_EDU_OPTIONS = ["None", "High School", "Some College", "Bachelor's", "Higher"]
PARENT_SUPPORT_OPTIONS = ["None", "Low", "Moderate", "High", "Very High"]

FEATURE_ORDER = [
    "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
    "Sports", "Music", "Volunteering", "GPA",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_frame(age, gender, ethnicity, parent_edu, study_time,
                       absences, gpa, tutoring, parent_support,
                       extracurricular, sports, music, volunteering):
    encoded = {
        "Age": age,
        "Gender": GENDER_OPTIONS.index(gender),
        "Ethnicity": ETHNICITY_OPTIONS.index(ethnicity),
        "ParentalEducation": PARENT_EDU_OPTIONS.index(parent_edu),
        "StudyTimeWeekly": study_time,
        "Absences": absences,
        "Tutoring": int(tutoring),
        "ParentalSupport": PARENT_SUPPORT_OPTIONS.index(parent_support),
        "Extracurricular": int(extracurricular),
        "Sports": int(sports),
        "Music": int(music),
        "Volunteering": int(volunteering),
        "GPA": gpa,
    }
    return pd.DataFrame([encoded])[FEATURE_ORDER]


def render_sidebar():
    st.sidebar.markdown("### Student Profile")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Personal**")
    age = st.sidebar.slider("Age", 15, 18, 16)
    gender = st.sidebar.selectbox("Gender", GENDER_OPTIONS)
    ethnicity = st.sidebar.selectbox("Ethnicity", ETHNICITY_OPTIONS)
    parent_edu = st.sidebar.selectbox("Parental Education", PARENT_EDU_OPTIONS)

    st.sidebar.markdown("**Academic**")
    study_time = st.sidebar.slider("Weekly Study Time (hrs)", 0.0, 20.0, 10.0, 0.5)
    absences = st.sidebar.slider("Absences (days/year)", 0, 30, 5)
    gpa = st.sidebar.slider("GPA", 0.0, 4.0, 2.5, 0.01)
    tutoring = st.sidebar.checkbox("Receives Tutoring")

    st.sidebar.markdown("**Activities**")
    parent_support = st.sidebar.selectbox("Parental Support Level", PARENT_SUPPORT_OPTIONS)
    extracurricular = st.sidebar.checkbox("Extracurricular Activities")
    sports = st.sidebar.checkbox("Sports")
    music = st.sidebar.checkbox("Music")
    volunteering = st.sidebar.checkbox("Volunteering")

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("Predict Grade", use_container_width=True)

    return {
        "age": age, "gender": gender, "ethnicity": ethnicity,
        "parent_edu": parent_edu, "study_time": study_time,
        "absences": absences, "gpa": gpa, "tutoring": tutoring,
        "parent_support": parent_support, "extracurricular": extracurricular,
        "sports": sports, "music": music, "volunteering": volunteering,
        "predict_clicked": predict_clicked,
    }


def render_summary_table(values):
    rows = {
        "Age": values["age"],
        "Gender": values["gender"],
        "Ethnicity": values["ethnicity"],
        "Parental Education": values["parent_edu"],
        "Weekly Study (hrs)": values["study_time"],
        "Absences": values["absences"],
        "GPA": f"{values['gpa']:.2f}",
        "Tutoring": "Yes" if values["tutoring"] else "No",
        "Parental Support": values["parent_support"],
        "Extracurricular": "Yes" if values["extracurricular"] else "No",
        "Sports": "Yes" if values["sports"] else "No",
        "Music": "Yes" if values["music"] else "No",
        "Volunteering": "Yes" if values["volunteering"] else "No",
    }
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(rows.items(), columns=["Field", "Value"]),
        hide_index=True,
        use_container_width=True,
    )


def render_prediction(model, input_frame):
    probabilities = model.predict_proba(input_frame)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_grade = GRADE_LABELS[predicted_index]
    confidence = probabilities[predicted_index] * 100
    color = GRADE_COLORS[predicted_grade]

    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card" style="background:{color}14; border-color:{color};">
        <div class="result-label">Predicted Grade</div>
        <div class="result-grade" style="color:{color};">{predicted_grade}</div>
        <div class="result-confidence">Confidence: <strong>{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    grades = [GRADE_LABELS[i] for i in range(5)]
    colors = [GRADE_COLORS[g] for g in grades]
    bars = ax.barh(grades, probabilities * 100, color=colors, height=0.55)

    for bar, value in zip(bars, probabilities * 100):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Grade Probability Distribution", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)


def render_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    highlight = {"GPA", "Absences", "StudyTimeWeekly"}
    colors = ["#618764" if name in highlight else "#9CB080" for name in importances.index]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    legend_handles = [
        mpatches.Patch(color="#618764", label="Strongest predictors"),
        mpatches.Patch(color="#9CB080", label="Other features"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)


def main():
    model = load_model()

    st.markdown('<div class="main-title">Student Grade Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Random Forest classifier trained on academic and behavioral data</div>',
        unsafe_allow_html=True,
    )

    values = render_sidebar()
    input_frame = build_input_frame(
        values["age"], values["gender"], values["ethnicity"], values["parent_edu"],
        values["study_time"], values["absences"], values["gpa"], values["tutoring"],
        values["parent_support"], values["extracurricular"], values["sports"],
        values["music"], values["volunteering"],
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        render_summary_table(values)

    with right_col:
        if values["predict_clicked"]:
            render_prediction(model, input_frame)
        else:
            st.info("Set the student profile in the sidebar, then click Predict Grade.")

    st.markdown("---")
    with st.expander("Model Feature Importance"):
        render_feature_importance(model, FEATURE_ORDER)

    st.markdown(
        '<div class="footer-text">Random Forest · Trained on 2,392 student records</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
