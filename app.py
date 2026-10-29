import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Titanic Data Analysis", page_icon="🚢", layout="wide")
sns.set_theme(style="whitegrid")

LABELS = {0: "Dead", 1: "Survived"}


@st.cache_data
def load_raw():
    return pd.read_csv("train.csv")


@st.cache_data
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Same steps as the notebook."""
    df = raw.drop_duplicates().copy()
    df = df.drop(columns="Cabin")                                   # 687/891 missing
    df["Age"] = df["Age"].fillna(df["Age"].mean())                   # mean imputation
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # mode imputation
    df["family_members"] = df["SibSp"] + df["Parch"]
    return df.drop(columns=["SibSp", "Parch"])


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def count_by(df, col, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=col, hue=df["Survived"].map(LABELS), ax=ax)
    ax.set_title(title)
    ax.legend(title="")
    show(fig)


def rate_table(df, col):
    t = (df.groupby(col)["Survived"].mean().mul(100).round(2)
           .rename("Survival rate (%)").reset_index())
    st.dataframe(t, hide_index=True)


raw = load_raw()
df = preprocess(raw)

st.title("🚢 Titanic Data Preprocessing & EDA")
st.caption("Interactive version of the analysis notebook (train.csv, 891 passengers).")

# ---- Sidebar filters -------------------------------------------------------
st.sidebar.header("Filters")
sex = st.sidebar.multiselect("Sex", sorted(df.Sex.unique()), default=sorted(df.Sex.unique()))
pclass = st.sidebar.multiselect("Passenger class", sorted(df.Pclass.unique()), default=sorted(df.Pclass.unique()))
emb = st.sidebar.multiselect("Embarked", sorted(df.Embarked.unique()), default=sorted(df.Embarked.unique()))
amin, amax = float(df.Age.min()), float(df.Age.max())
age = st.sidebar.slider("Age", amin, amax, (amin, amax))

f = df[df.Sex.isin(sex) & df.Pclass.isin(pclass) & df.Embarked.isin(emb) & df.Age.between(*age)]
st.sidebar.write(f"**{len(f)}** of {len(df)} passengers selected")

if f.empty:
    st.warning("No passengers match the filters.")
    st.stop()

tab_data, tab_prep, tab_surv, tab_fare, tab_age = st.tabs(
    ["Dataset", "Preprocessing", "Survival", "Fare", "Age & Family"]
)

# ---- Dataset ---------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers (filtered)", len(f))
    c2.metric("Survival rate", f"{f.Survived.mean() * 100:.1f}%")
    c3.metric("Average fare", f"{f.Fare.mean():.2f}")
    st.subheader("Data preview")
    st.dataframe(f.head(50), hide_index=True)
    st.subheader("Summary statistics")
    st.dataframe(f.describe())

# ---- Preprocessing ---------------------------------------------------------
with tab_prep:
    st.subheader("Duplicates")
    st.write(f"Duplicate rows found: **{raw.duplicated().sum()}**")
    st.subheader("Missing values (raw data)")
    miss = raw.isnull().sum().rename("Missing").to_frame()
    miss["% of rows"] = (miss["Missing"] / len(raw) * 100).round(1)
    st.dataframe(miss[miss.Missing > 0])
    st.markdown(
        """
- **Cabin** has 687 of 891 values missing, so the column is dropped.
- **Age** has 177 missing values, filled with the mean age.
- **Embarked** has 2 missing values, filled with the mode (`S`).
- **SibSp + Parch** are combined into a single `family_members` column.
"""
    )
    st.subheader("Age distribution (after cleaning)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.kdeplot(df.Age, ax=ax, fill=True)
    show(fig)

# ---- Survival --------------------------------------------------------------
with tab_surv:
    st.subheader("Overall survival")
    counts = f.Survived.value_counts()
    c1, c2 = st.columns(2)
    with c1:
        count_by(f.assign(All="All passengers"), "All", "Survived vs dead")
    with c2:
        st.metric("Survivors", f"{counts.get(1, 0)}  ({counts.get(1, 0) / len(f) * 100:.2f}%)")
        st.metric("Dead", f"{counts.get(0, 0)}  ({counts.get(0, 0) / len(f) * 100:.2f}%)")

    for col, name, note in [
        ("Sex", "Gender", "Women survived at a far higher rate than men."),
        ("Pclass", "Passenger class", "Survival drops steadily from 1st to 3rd class."),
        ("Embarked", "Port of embarkation", "Cherbourg (C) passengers survived most often."),
    ]:
        st.subheader(f"Survival by {name}")
        c1, c2 = st.columns([2, 1])
        with c1:
            count_by(f, col, f"Survival by {name}")
        with c2:
            rate_table(f, col)
            st.info(note + " (Insight is based on the full dataset.)")

    st.subheader("Survival rate by class (pie charts)")
    classes = sorted(f.Pclass.unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4), squeeze=False)
    for ax, c in zip(axes[0], classes):
        s = f[f.Pclass == c].Survived.value_counts().reindex([0, 1], fill_value=0)
        ax.pie(s, labels=["Dead", "Survived"], autopct="%1.1f%%", explode=[0, 0.1])
        ax.set_title(f"Pclass {c}")
    show(fig)

# ---- Fare ------------------------------------------------------------------
with tab_fare:
    st.subheader("Fare distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(f.Fare, ax=ax, fill=True, clip=(0, None))
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=f.Fare, ax=ax)
        show(fig)

    st.subheader("Fare vs other columns")
    by = st.selectbox("Compare fare by", ["Pclass", "Survived", "Sex", "Embarked"])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=f, x=by, y="Fare", ax=ax)
        ax.set_title("Mean fare")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=f, x=by, y="Fare", ax=ax)
        ax.set_title("Fare spread")
        show(fig)

    st.subheader("Outliers")
    q1, q3 = df.Fare.quantile([0.25, 0.75])
    iqr = q3 - q1
    z = (df.Fare - df.Fare.mean()) / df.Fare.std()
    c1, c2 = st.columns(2)
    c1.metric("IQR outliers (Fare > Q3 + 3×IQR)", int((df.Fare > q3 + 3 * iqr).sum()))
    c2.metric("Z-score outliers (|z| > 2)", int((z.abs() > 2).sum()))
    st.write("Passengers who paid more than 500:")
    st.dataframe(df[df.Fare > 500], hide_index=True)

# ---- Age & family ----------------------------------------------------------
with tab_age:
    st.subheader("Age vs other columns")
    by = st.selectbox("Compare age by", ["Pclass", "Survived", "Sex"], key="age_by")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=f, x=by, y="Age", ax=ax)
        ax.set_title("Mean age")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=f, x=by, y="Age", ax=ax)
        ax.set_title("Age spread")
        show(fig)

    st.subheader("Survival by family size")
    count_by(f, "family_members", "Survival by number of family members aboard")
    rate_table(f, "family_members")
