"""
AI-Driven Store Footfall Impact Analysis on Sales
Stage 5 - Streamlit Dashboard

Run with:  streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

st.set_page_config(page_title="Footfall Impact on Sales", layout="wide")


# ----------------------------------------------------------------------
# LOAD DATA & MODEL (cached so it only happens once)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_footfall_sales.csv", parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    model = joblib.load("models/model2_multiple_regression.pkl")
    with open("models/model_summary.json") as f:
        summary = json.load(f)
    return model, summary


@st.cache_data
def load_store_impact():
    return pd.read_csv("models/store_level_impact.csv")


df = load_data()
model2, summary = load_model()
store_impact_df = load_store_impact()


# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------
st.sidebar.header("Filters")

stores = sorted(df["store"].unique())
selected_stores = st.sidebar.multiselect(
    "Select Store(s)", options=stores, default=stores
)

min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

weekend_filter = st.sidebar.selectbox(
    "Day Type", options=["All", "Weekday only", "Weekend only"]
)

# Apply filters
mask = df["store"].isin(selected_stores)
if len(date_range) == 2:
    mask &= (df["date"] >= pd.Timestamp(date_range[0])) & \
            (df["date"] <= pd.Timestamp(date_range[1]))
if weekend_filter == "Weekday only":
    mask &= df["is_weekend"] == 0
elif weekend_filter == "Weekend only":
    mask &= df["is_weekend"] == 1

fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the selected filters. Please widen your selection.")
    st.stop()


# ----------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------
st.title("AI-Driven Store Footfall Impact Analysis on Sales")
st.caption("Dominick's Finer Foods dataset | Kilts Center, University of Chicago")

tabs = st.tabs([
    "Overview", "Footfall Analysis", "Sales Analysis",
    "Footfall vs Sales", "Regression Model", "Impact Analysis", "Insights"
])


# ----------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------
with tabs[0]:
    st.subheader("Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"${fdf['Total_Sales'].sum():,.0f}")
    c2.metric("Avg Daily Sales", f"${fdf['Total_Sales'].mean():,.0f}")
    c3.metric("Total Footfall", f"{fdf['custcoun'].sum():,.0f}")
    c4.metric("Avg Daily Footfall", f"{fdf['custcoun'].mean():,.0f}")
    c5.metric("Stores / Records", f"{fdf['store'].nunique()} / {len(fdf):,}")

    st.markdown("---")
    st.write("Sample of the filtered data:")
    st.dataframe(fdf.head(20), use_container_width=True)


# ----------------------------------------------------------------------
# TAB 2: FOOTFALL ANALYSIS
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Footfall Analysis")

    daily_ff = fdf.groupby("date")["custcoun"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily_ff.plot(ax=ax, color="steelblue")
    ax.set_title("Footfall Over Time")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        top_stores = fdf.groupby("store")["custcoun"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        top_stores.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Top 15 Stores by Avg Footfall")
        ax.invert_yaxis()
        st.pyplot(fig)

    with col2:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_ff = fdf.groupby("day_of_week")["custcoun"].mean().reindex(dow_order)
        fig, ax = plt.subplots(figsize=(6, 5))
        dow_ff.plot(kind="bar", ax=ax, color="teal")
        ax.set_title("Avg Footfall by Day of Week")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)


# ----------------------------------------------------------------------
# TAB 3: SALES ANALYSIS
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Sales Analysis")

    daily_sales = fdf.groupby("date")["Total_Sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily_sales.plot(ax=ax, color="darkorange")
    ax.set_title("Sales Over Time")
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        top_sales = fdf.groupby("store")["Total_Sales"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        top_sales.plot(kind="barh", ax=ax, color="darkorange")
        ax.set_title("Top 15 Stores by Avg Sales")
        ax.invert_yaxis()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(fdf["Total_Sales"], bins=60, kde=True, ax=ax, color="darkorange")
        ax.set_title("Sales Distribution")
        st.pyplot(fig)


# ----------------------------------------------------------------------
# TAB 4: FOOTFALL VS SALES
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Footfall vs Sales")

    sample = fdf.sample(min(3000, len(fdf)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(data=sample, x="custcoun", y="Total_Sales", ax=ax,
                scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "red"})
    ax.set_title("Footfall vs Sales with Regression Line")
    ax.set_xlabel("Customer Count")
    ax.set_ylabel("Total Sales ($)")
    st.pyplot(fig)

    corr = fdf["custcoun"].corr(fdf["Total_Sales"])
    st.metric("Correlation (Footfall vs Sales)", f"{corr:.3f}")

    corr_cols = ["custcoun", "Total_Sales", "Sales_Per_Customer", "is_weekend"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(fdf[corr_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


# ----------------------------------------------------------------------
# TAB 5: REGRESSION MODEL
# ----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Regression Model (Model 2: Multiple Linear Regression)")

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{summary['model2_r2']:.4f}")
    c2.metric("MAE", f"${summary['model2_mae']:,.2f}")
    c3.metric("RMSE", f"${summary['model2_rmse']:,.2f}")

    st.markdown("---")
    st.write("**Model 1 (Simple Linear Regression) for comparison:**")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{summary['model1_r2']:.4f}")
    c2.metric("MAE", f"${summary['model1_mae']:,.2f}")
    c3.metric("RMSE", f"${summary['model1_rmse']:,.2f}")

    st.markdown("---")
    st.write(f"**Footfall coefficient (Model 2):** ${summary['model2_coef_footfall']:.2f} "
              f"in sales per additional customer.")

    # Actual vs predicted on the filtered subset
    dow_dummies = pd.get_dummies(fdf["day_of_week"], drop_first=True)
    for col in ["Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]:
        if col not in dow_dummies.columns:
            dow_dummies[col] = 0
    features = pd.concat([fdf[["custcoun", "is_weekend", "month"]], dow_dummies], axis=1)
    features = features[model2.feature_names_in_]  # match training column order

    preds = model2.predict(features)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fdf["Total_Sales"], preds, alpha=0.2, s=10, color="steelblue")
    lims = [fdf["Total_Sales"].min(), fdf["Total_Sales"].max()]
    ax.plot(lims, lims, color="red", linewidth=1.5)
    ax.set_xlabel("Actual Sales ($)")
    ax.set_ylabel("Predicted Sales ($)")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)


# ----------------------------------------------------------------------
# TAB 6: IMPACT ANALYSIS
# ----------------------------------------------------------------------
with tabs[5]:
    st.subheader("Impact Analysis")

    coef = summary["model2_coef_footfall"]
    impact_100 = coef * 100
    avg_sales = fdf["Total_Sales"].mean()
    pct_impact = (impact_100 / avg_sales) * 100

    st.info(
        f"**If footfall increases by 100 customers, the model estimates an average "
        f"change of ${impact_100:,.2f} in sales, holding day-of-week and month constant.**\n\n"
        f"That is roughly **{pct_impact:.2f}%** of an average store-day's revenue "
        f"(${avg_sales:,.0f})."
    )

    st.markdown("---")
    st.write("**Impact varies by store** (sales gained per additional customer):")

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = store_impact_df.sort_values("coef_per_customer", ascending=False)
    ax.bar(range(len(plot_df)), plot_df["coef_per_customer"], color="steelblue")
    ax.axhline(coef, color="red", linestyle="--", label=f"Overall model: ${coef:.2f}")
    ax.set_xlabel("Store (sorted)")
    ax.set_ylabel("Sales per extra customer ($)")
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.write("**Top 5 highest-converting stores**")
    col1.dataframe(plot_df.head(5)[["store", "coef_per_customer", "avg_footfall", "avg_sales"]],
                    hide_index=True)
    col2.write("**Bottom 5 lowest-converting stores**")
    col2.dataframe(plot_df.tail(5)[["store", "coef_per_customer", "avg_footfall", "avg_sales"]],
                    hide_index=True)


# ----------------------------------------------------------------------
# TAB 7: INSIGHTS (auto-generated from the filtered data)
# ----------------------------------------------------------------------
with tabs[6]:
    st.subheader("Insights")

    best_day = fdf.groupby("day_of_week")["Total_Sales"].mean().idxmax()
    worst_day = fdf.groupby("day_of_week")["Total_Sales"].mean().idxmin()
    best_store = fdf.groupby("store")["Total_Sales"].mean().idxmax()
    corr_val = fdf["custcoun"].corr(fdf["Total_Sales"])

    insights = [
        f"Footfall and sales are strongly correlated (r = {corr_val:.2f}) across the selected data.",
        f"**{best_day}** has the highest average sales; **{worst_day}** has the lowest.",
        f"Store **{best_store}** has the highest average daily sales in the current selection.",
        f"Each additional customer is associated with **${coef:.2f}** more in sales on average.",
        f"100 extra customers translate to roughly **${impact_100:,.0f}** in additional daily sales "
        f"({pct_impact:.2f}% of an average day's revenue).",
        f"The multiple regression model (R² = {summary['model2_r2']:.3f}) explains "
        f"{summary['model2_r2']*100:.1f}% of the variation in daily sales using footfall and "
        f"seasonality alone.",
    ]

    for i, point in enumerate(insights, 1):
        st.markdown(f"{i}. {point}")