import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Seizure Classification",
    layout="wide",
)

st.title("EEG Seizure Classification")
st.markdown(
    "Upload the **Epileptic Seizure Recognition** CSV to explore, train, "
    "and evaluate Logistic Regression and Random Forest classifiers."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader(
        "Upload Data (.csv)", type=["csv"],
        help="Upload the spreadsheet with the brain wave recordings and seizure labels."
    )
    st.markdown("---")
    test_size      = st.slider(
        "Test Data Size",
        0.10, 0.40, 0.20, 0.05,
        help="Percentage of data set aside to test the model after training. 0.20 means 20% is used for testing."
    )
    random_state   = st.number_input(
        "Shuffle Seed",
        value=42, step=1,
        help="Keeps the data mix consistent each time you run the app. Leave as 42 for repeatable results."
    )
    n_estimators   = st.slider(
        "Forest Size (Random Forest)",
        50, 500, 100, 50,
        help="How many decision paths to combine. More paths increase accuracy but slow down training."
    )
    lr_max_iter    = st.slider(
        "Max Training Steps (Log. Reg.)",
        200, 2000, 1000, 100,
        help="Limits how many times the model can adjust itself. Increase this if the training stops early."
    )
    top_n_features = st.slider(
        "Top Signals to Show",
        5, 30, 20, 5,
        help="How many of the most important brain wave moments to display in the final charts."
    )
    run_btn        = st.button("Run Analysis", use_container_width=True)
# ── Cached training ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train(file_bytes, test_size, random_state, n_estimators, lr_max_iter):
    df = pd.read_csv(io.BytesIO(file_bytes))

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["y"] = df["y"].apply(lambda x: 1 if x == 1 else 0)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=lr_max_iter)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    return df, X, y, X_test, y_test, lr, rf, y_pred_lr, y_pred_rf, scaler

# ── Guard: require upload and button press ─────────────────────────────────────
if uploaded_file is None:
    st.info("Upload the CSV in the sidebar to get started.")
    st.stop()

if not run_btn:
    st.info("Press Run Analysis in the sidebar to train the models.")
    st.stop()

with st.spinner("Training models — this may take a moment..."):
    (df, X, y, X_test, y_test,
     lr, rf, y_pred_lr, y_pred_rf, scaler) = load_and_train(
        uploaded_file.read(),
        test_size, int(random_state), n_estimators, lr_max_iter
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview",
    "EEG Signals",
    "Model Performance",
    "Feature Importance",
    "Live Prediction",
])

# ── Tab 1: Data Overview ───────────────────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", f"{len(df):,}")
    c2.metric("Features", f"{X.shape[1]:,}")
    c3.metric("Seizure rate", f"{y.mean() * 100:.1f}%")

    st.subheader("Sample data (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = y.value_counts().sort_index()
        ax.bar(
            ["No Seizure (0)", "Seizure (1)"],
            counts.values,
            color=["steelblue", "crimson"],
        )
        ax.set_ylabel("Count")
        ax.set_title("Seizure vs No Seizure")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Descriptive statistics")
        st.dataframe(
            df.describe().T[["mean", "std", "min", "max"]],
            use_container_width=True,
        )

# ── Tab 2: EEG Signals ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Representative EEG waveforms")
    seizure_signal = df[df["y"] == 1].iloc[0, :-1].to_numpy().astype(float)
    normal_signal  = df[df["y"] == 0].iloc[0, :-1].to_numpy().astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(seizure_signal, color="crimson")
    axes[0].set_title("Seizure EEG Signal")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(normal_signal, color="steelblue")
    axes[1].set_title("Non-Seizure EEG Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time Points")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 3: Model Performance ──────────────────────────────────────────────────
with tab3:
    lr_acc = accuracy_score(y_test, y_pred_lr)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    st.subheader("Accuracy comparison")
    ca, cb = st.columns(2)
    ca.metric("Logistic Regression", f"{lr_acc * 100:.2f}%")
    cb.metric("Random Forest", f"{rf_acc * 100:.2f}%",
              delta=f"{(rf_acc - lr_acc) * 100:+.2f}% vs LR")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest — Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_rf),
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Seizure", "Seizure"],
                    yticklabels=["No Seizure", "Seizure"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")
        report_rf = classification_report(
            y_test, y_pred_rf,
            target_names=["No Seizure", "Seizure"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report_rf).T, use_container_width=True)

    with col2:
        st.subheader("Logistic Regression — Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_lr),
                    annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["No Seizure", "Seizure"],
                    yticklabels=["No Seizure", "Seizure"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")
        report_lr = classification_report(
            y_test, y_pred_lr,
            target_names=["No Seizure", "Seizure"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report_lr).T, use_container_width=True)

# ── Tab 4: Feature Importance ─────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Random Forest — Top {top_n_features} Features")
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_rf = importances.sort_values(ascending=False).head(top_n_features)
        fig, ax = plt.subplots(figsize=(10, 4))
        top_rf.plot(kind="bar", color="steelblue", ax=ax)
        ax.set_title(f"Top {top_n_features} EEG Time Points (RF)")
        ax.set_xlabel("EEG Time Point")
        ax.set_ylabel("Importance Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            top_rf.rename("importance").reset_index().rename(columns={"index": "feature"}),
            use_container_width=True,
        )

    with col2:
        st.subheader(f"Logistic Regression — Top {top_n_features} Features")
        coefs  = pd.Series(lr.coef_[0], index=X.columns)
        top_lr = coefs.abs().sort_values(ascending=False).head(top_n_features)
        fig, ax = plt.subplots(figsize=(10, 4))
        top_lr.plot(kind="bar", color="crimson", ax=ax)
        ax.set_title(f"Top {top_n_features} Features — Absolute LR Coefficient")
        ax.set_xlabel("EEG Time Point")
        ax.set_ylabel("Absolute Coefficient")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            top_lr.rename("abs_coef").reset_index().rename(columns={"index": "feature"}),
            use_container_width=True,
        )

# ── Tab 5: Live Prediction ────────────────────────────────────────────────────
with tab5:
    st.subheader("Predict from a raw EEG row")
    st.markdown(
        f"Paste {X.shape[1]} comma-separated EEG amplitude values "
        "(one complete time-series row), then click **Predict**."
    )

    raw_input = st.text_area(
        "EEG values (comma-separated)",
        placeholder=f"e.g. 135, -77, 207, ... ({X.shape[1]} numbers total)",
        height=120,
    )

    model_choice = st.radio(
        "Model", ["Random Forest", "Logistic Regression"], horizontal=True
    )

    if st.button("Predict"):
        try:
            values = [float(v.strip()) for v in raw_input.split(",")]
            if len(values) != X.shape[1]:
                st.error(
                    f"Expected {X.shape[1]} values, got {len(values)}. "
                    "Please paste a complete row."
                )
            else:
                arr        = np.array(values).reshape(1, -1)
                arr_scaled = scaler.transform(arr)
                model      = rf if model_choice == "Random Forest" else lr
                pred       = model.predict(arr_scaled)[0]
                proba      = model.predict_proba(arr_scaled)[0]

                label = "SEIZURE DETECTED" if pred == 1 else "No Seizure"
                st.subheader(f"Prediction: {label}")

                col_p, col_q = st.columns(2)
                col_p.metric("P(No Seizure)", f"{proba[0] * 100:.1f}%")
                col_q.metric("P(Seizure)",    f"{proba[1] * 100:.1f}%")

                fig, ax = plt.subplots(figsize=(14, 3))
                color = "crimson" if pred == 1 else "steelblue"
                ax.plot(values, color=color)
                ax.set_title(f"Input EEG Waveform — {label}")
                ax.set_xlabel("Time Points")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
                plt.close(fig)

        except ValueError:
            st.error(
                "Could not parse the input. "
                "Make sure all values are numbers separated by commas."
            )

    st.markdown("---")
    st.caption(
        "Copy any row from the dataset (excluding the y column) "
        "and paste it above to test a real sample."
    )