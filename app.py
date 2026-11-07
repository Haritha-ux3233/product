
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Product Quality Intelligence",
    page_icon="🏭",
    layout="wide"
)

# ------------------------------------------------
# LOAD CSS
# ------------------------------------------------

def load_css():
    if os.path.exists("style.css"):
        with open("style.css", "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------

@st.cache_resource
def load_model_files():

    model = joblib.load("product_quality_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return model, label_encoder


try:
    model, label_encoder = load_model_files()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.warning(
        "Keep product_quality_model.pkl and label_encoder.pkl "
        "in the same folder as app.py."
    )
    st.stop()


# Use the exact feature names and order the model was trained with
selected_features = list(model.feature_names_in_)
class_names = list(label_encoder.classes_)

# ------------------------------------------------
# HISTORY
# ------------------------------------------------

HISTORY_FILE = "history.csv"


def get_history():

    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()


def save_history(values, prediction, probabilities):

    record = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Prediction": str(prediction)
    }

    for feature, value in values.items():
        record[feature] = value

    for i, name in enumerate(class_names):
        record[f"{name} Probability (%)"] = round(
            float(probabilities[i]) * 100, 3
        )

    new_record = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        new_record = pd.concat(
            [old, new_record],
            ignore_index=True
        )

    new_record.to_csv(HISTORY_FILE, index=False)


# ------------------------------------------------
# METRIC CARD
# ------------------------------------------------

def metric_card(title, value, description=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.markdown("# 🏭 ProductIQ")
st.sidebar.markdown("### Quality Assessment System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "New Prediction",
        "Bayesian Analysis",
        "Gaussian Analysis",
        "Prediction History",
        "Feature Analysis",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Gaussian Naive Bayes")
st.sidebar.markdown("**Classes:** LOW / MEDIUM / HIGH")
st.sidebar.markdown(
    f"**Features:** {len(selected_features)}"
)


# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-heading">Product Quality Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">'
        'Monitor product quality predictions and probability analysis'
        '</div>',
        unsafe_allow_html=True
    )

    history = get_history()

    total = len(history)

    if total > 0:
        counts = history["Prediction"].astype(str).str.upper().value_counts()
    else:
        counts = pd.Series(dtype=int)

    low = int(counts.get("LOW", 0))
    medium = int(counts.get("MEDIUM", 0))
    high = int(counts.get("HIGH", 0))

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Predictions", total, "All saved predictions")

    with c2:
        metric_card("LOW Quality", low, "Low quality predictions")

    with c3:
        metric_card("MEDIUM Quality", medium, "Medium quality predictions")

    with c4:
        metric_card("HIGH Quality", high, "High quality predictions")

    st.write("")

    left, right = st.columns([1, 1])

    with left:

        st.markdown("### Quality Distribution")

        if total > 0:

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.bar(
                ["LOW", "MEDIUM", "HIGH"],
                [low, medium, high]
            )

            ax.set_ylabel("Number of Predictions")
            ax.set_title("Predicted Quality Classes")
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("No predictions yet. Start with New Prediction.")

    with right:

        st.markdown("### Model Information")

        st.markdown(
            f"""
            <div class="info-card">
                <p><b>Algorithm:</b> Gaussian Naive Bayes</p>
                <p><b>Feature Count:</b> {len(selected_features)}</p>
                <p><b>Classification:</b> Three Classes</p>
                <p><b>Probability:</b> Bayesian Posterior</p>
                <p><b>Distribution:</b> Gaussian</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Recent Predictions")

    if total > 0:
        st.dataframe(
            history.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Prediction history will appear here.")


# ------------------------------------------------
# NEW PREDICTION
# ------------------------------------------------

elif page == "New Prediction":

    st.markdown(
        '<div class="main-heading">New Product Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Enter values for the {len(selected_features)} features used during model training."
    )

    st.info(
        "Enter the same type of values and use the same feature order "
        "as your training data."
    )

    values = {}

    with st.form("prediction_form"):

        st.markdown("### Product Input Details")

        col1, col2 = st.columns(2)

        for i, feature in enumerate(selected_features):

            with col1 if i % 2 == 0 else col2:

                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.4f",
                    key=f"feature_{i}"
                )

        submit = st.form_submit_button(
            "Predict Quality Level",
            use_container_width=True
        )

    if submit:

        try:

            input_df = pd.DataFrame(
                [[values[f] for f in selected_features]],
                columns=selected_features
            )

            prediction_encoded = model.predict(input_df)[0]

            prediction = label_encoder.inverse_transform(
                [prediction_encoded]
            )[0]

            probabilities = model.predict_proba(input_df)[0]

            st.session_state["last_input"] = values.copy()
            st.session_state["last_prediction"] = prediction
            st.session_state["last_probabilities"] = probabilities.copy()

            save_history(values, prediction, probabilities)

            st.markdown("---")
            st.markdown("### Prediction Result")

            prediction_upper = str(prediction).upper()

            if prediction_upper == "LOW":
                st.error(f"Predicted Quality: {prediction}")

            elif prediction_upper == "MEDIUM":
                st.warning(f"Predicted Quality: {prediction}")

            elif prediction_upper == "HIGH":
                st.success(f"Predicted Quality: {prediction}")

            else:
                st.success(f"Predicted Class: {prediction}")

            st.markdown("### Bayesian Class Probabilities")

            probability_columns = st.columns(len(class_names))

            for i, class_name in enumerate(class_names):

                with probability_columns[i]:
                    metric_card(
                        class_name,
                        f"{probabilities[i] * 100:.2f}%",
                        "Posterior probability"
                    )

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                class_names,
                probabilities * 100
            )

            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Gaussian Naive Bayes Prediction")
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

            st.success("Prediction saved successfully.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ------------------------------------------------
# BAYESIAN ANALYSIS
# ------------------------------------------------

elif page == "Bayesian Analysis":

    st.markdown(
        '<div class="main-heading">Bayesian Probability Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Bayes' Theorem")

    st.latex(
        r"P(C|X)=\frac{P(X|C)P(C)}{P(X)}"
    )

    st.markdown("""
    **Explanation**

    - P(C|X): Posterior probability of a class.
    - P(X|C): Likelihood of the observed features for a class.
    - P(C): Prior probability of a class.
    - P(X): Evidence or marginal probability.
    """)

    st.markdown("### Class Prior Probabilities")

    prior_df = pd.DataFrame({
        "Class": class_names,
        "Prior Probability": model.class_prior_,
        "Prior Percentage": model.class_prior_ * 100
    })

    st.dataframe(
        prior_df.style.format({
            "Prior Probability": "{:.6f}",
            "Prior Percentage": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    if "last_probabilities" in st.session_state:

        st.markdown("### Latest Posterior Probabilities")

        posterior_df = pd.DataFrame({
            "Class": class_names,
            "Posterior Probability": st.session_state["last_probabilities"],
            "Posterior Percentage":
                st.session_state["last_probabilities"] * 100
        })

        st.dataframe(
            posterior_df.style.format({
                "Posterior Probability": "{:.6f}",
                "Posterior Percentage": "{:.3f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Make a prediction to see its posterior probabilities.")


# ------------------------------------------------
# GAUSSIAN ANALYSIS
# ------------------------------------------------

elif page == "Gaussian Analysis":

    st.markdown(
        '<div class="main-heading">Gaussian Distribution Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Gaussian Probability Density Equation")

    st.latex(
        r"f(x|\mu,\sigma^2)="
        r"\frac{1}{\sqrt{2\pi\sigma^2}}"
        r"e^{-\frac{(x-\mu)^2}{2\sigma^2}}"
    )

    st.markdown("""
    **Mathematical terms**

    - x: Input feature value.
    - μ: Class-specific mean.
    - σ²: Class-specific variance.
    - π: Mathematical constant.
    - e: Exponential constant.
    """)

    if "last_input" in st.session_state:

        values = st.session_state["last_input"]
        prediction = st.session_state["last_prediction"]

        encoded_class = label_encoder.transform([prediction])[0]
        class_index = list(model.classes_).index(encoded_class)

        rows = []

        for i, feature in enumerate(selected_features):

            x = float(values[feature])
            mean = float(model.theta_[class_index, i])
            variance = max(float(model.var_[class_index, i]), 1e-12)

            density = (
                1 / np.sqrt(2 * np.pi * variance)
            ) * np.exp(
                -((x - mean) ** 2) / (2 * variance)
            )

            rows.append({
                "Feature": feature,
                "Input Value": x,
                "Mean": mean,
                "Variance": variance,
                "Gaussian Density": density
            })

        gaussian_df = pd.DataFrame(rows)

        st.markdown(
            f"### Gaussian calculations for predicted class: {prediction}"
        )

        st.dataframe(
            gaussian_df.style.format({
                "Input Value": "{:.4f}",
                "Mean": "{:.4f}",
                "Variance": "{:.6f}",
                "Gaussian Density": "{:.8g}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Each density is a feature-wise Gaussian likelihood. "
            "The model combines feature likelihoods and class priors "
            "to calculate posterior class probabilities."
        )

    else:
        st.info("Make a prediction first to view Gaussian calculations.")


# ------------------------------------------------
# PREDICTION HISTORY
# ------------------------------------------------

elif page == "Prediction History":

    st.markdown(
        '<div class="main-heading">Prediction History</div>',
        unsafe_allow_html=True
    )

    history = get_history()

    if not history.empty:

        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="product_quality_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            os.remove(HISTORY_FILE)
            st.success("History cleared.")
            st.rerun()

    else:
        st.info("No prediction records available.")


# ------------------------------------------------
# FEATURE ANALYSIS
# ------------------------------------------------

elif page == "Feature Analysis":

    st.markdown(
        '<div class="main-heading">Feature Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("The following features were used to train the model.")

    st.dataframe(
        pd.DataFrame({
            "Feature Number": range(1, len(selected_features) + 1),
            "Selected Feature": selected_features
        }),
        use_container_width=True,
        hide_index=True
    )

    # Show Gaussian class means by feature
    mean_rows = []

    for class_index, class_value in enumerate(model.classes_):

        class_name = label_encoder.inverse_transform(
            [class_value]
        )[0]

        for i, feature in enumerate(selected_features):

            mean_rows.append({
                "Class": class_name,
                "Feature": feature,
                "Mean": model.theta_[class_index, i],
                "Variance": model.var_[class_index, i]
            })

    st.markdown("### Class-wise Gaussian Parameters")

    st.dataframe(
        pd.DataFrame(mean_rows),
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------
# ABOUT
# ------------------------------------------------

elif page == "About":

    st.markdown(
        '<div class="main-heading">About ProductIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Machine Learning-Based Product Quality and Defect Risk Assessment

    This application uses Gaussian Naive Bayes to classify product
    quality into LOW, MEDIUM and HIGH categories.

    **Main mathematical concepts**

    1. Bayes' theorem
    2. Prior probability
    3. Conditional probability
    4. Gaussian probability density function
    5. Posterior probability

    **Machine learning workflow**

    Dataset → Feature Selection → Gaussian Naive Bayes →
    Probability Calculation → Quality Prediction

    **Technology**

    Python, Pandas, NumPy, Scikit-learn, Matplotlib and Streamlit.
    """)
