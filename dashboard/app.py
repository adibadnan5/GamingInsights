from pathlib import Path
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


from supabase import create_client

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="GamingInsights | Gaming Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gaming_mental_health_spark_cleaned.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gaming_addiction_prediction_model_compatible.joblib"
)


# ---------------------------------------------------------
# Supabase Configuration
# ---------------------------------------------------------

def get_app_secret(name):
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


SUPABASE_URL = get_app_secret("SUPABASE_URL")
SUPABASE_KEY = get_app_secret("SUPABASE_KEY")

PREDICTION_TABLE = "prediction_history"


@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Supabase credentials are missing. Configure SUPABASE_URL "
            "and SUPABASE_KEY in Streamlit Secrets or environment variables."
        )

    return create_client(SUPABASE_URL, SUPABASE_KEY)


try:
    supabase_client = get_supabase_client()
    supabase_connected = True

except Exception as error:
    supabase_connected = False
    supabase_client = None
    st.warning(f"Supabase connection unavailable: {error}")

# ---------------------------------------------------------
# Retrieve live prediction inputs from Supabase
# ---------------------------------------------------------

def load_live_predictions():
    if not supabase_connected or supabase_client is None:
        return pd.DataFrame()

    try:
        response = (
            supabase_client
            .table(PREDICTION_TABLE)
            .select("input_features, predicted_score, addiction_category, created_at")
            .order("created_at", desc=True)
            .limit(5000)
            .execute()
        )
        records = response.data or []

        if not records:
            return pd.DataFrame()

        live_df = pd.json_normalize(records)
        rename_map = {
            f"input_features.{feature}": feature
            for feature in [
                "daily_gaming_hours",
                "weekly_sessions",
                "night_gaming_ratio",
                "toxic_exposure",
                "sleep_hours",
                "exercise_hours",
                "screen_time_total"
            ]
        }
        return live_df.rename(columns=rename_map)

    except Exception as error:
        st.warning(f"Unable to retrieve live predictions from Supabase: {error}")
        return pd.DataFrame()


# ---------------------------------------------------------
# Load resources
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model_package = load_model_package()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.stop()


model = model_package["model"]
model_features = model_package["features"]
model_name = model_package["model_name"]


# Use a sample for interactive charts
chart_sample_size = min(100_000, len(df))

df_chart = df.sample(
    n=chart_sample_size,
    random_state=42
)

# Retrieve saved prediction inputs and create a separate live analytics layer.
live_predictions_df = load_live_predictions()

live_feature_columns = [
    "daily_gaming_hours",
    "weekly_sessions",
    "night_gaming_ratio",
    "toxic_exposure",
    "sleep_hours",
    "exercise_hours",
    "screen_time_total"
]

if not live_predictions_df.empty:
    available_live_features = [
        column for column in live_feature_columns
        if column in live_predictions_df.columns
    ]

    live_gaming_df = live_predictions_df[
        available_live_features
    ].copy()

    # Derive the category/score fields used by the gaming charts.
    if "predicted_score" in live_predictions_df.columns:
        live_gaming_df["addiction_level"] = (
            live_predictions_df["predicted_score"]
        )

    if "addiction_category" in live_predictions_df.columns:
        live_gaming_df["addiction_category"] = (
            live_predictions_df["addiction_category"]
        )

    # Keep only records with a usable value for at least one feature.
    if not live_gaming_df.empty:
        df_gaming_combined = pd.concat(
            [df, live_gaming_df],
            ignore_index=True,
            sort=False
        )
        df_chart_combined = pd.concat(
            [df_chart, live_gaming_df],
            ignore_index=True,
            sort=False
        )
    else:
        df_gaming_combined = df.copy()
        df_chart_combined = df_chart.copy()
else:
    df_gaming_combined = df.copy()
    df_chart_combined = df_chart.copy()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def convert_addiction_category(score):
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    else:
        return "High"


def category_colour(category):
    colours = {
        "Low": "#2ECC71",
        "Moderate": "#F1C40F",
        "High": "#E74C3C"
    }

    return colours.get(category, "#3366CC")


def format_variable_name(variable):
    return variable.replace("_", " ").title()




def generate_prescriptive_recommendations(
    daily_gaming_hours,
    weekly_sessions,
    night_gaming_ratio,
    toxic_exposure,
    sleep_hours,
    exercise_hours,
    screen_time_total
):
    """
    Generate recommendations only for input factors that cross
    the project's initial rule-based review thresholds.

    These are configurable project rules, not clinical cut-offs.
    """
    recommendations = []

    # Gaming duration: prioritise reducing unusually long daily play.
    if daily_gaming_hours >= 8:
        recommendations.append({
            "factor": "Daily gaming hours",
            "current": f"{daily_gaming_hours:.1f} hours/day",
            "advice": (
                "Your daily gaming duration is high. Gradually reduce "
                "your gaming time towards a suggested target of 3 hours "
                "per day. You could first reduce it to 5 hours, then "
                "work towards the 3-hour target."
            )
        })
    elif daily_gaming_hours >= 6:
        recommendations.append({
            "factor": "Daily gaming hours",
            "current": f"{daily_gaming_hours:.1f} hours/day",
            "advice": (
                "Your daily gaming duration is elevated. Try reducing "
                "your gaming time by around 1 hour and schedule "
                "non-gaming activities."
            )
        })

    # Repeated sessions may indicate frequent gaming throughout the week.
    if weekly_sessions >= 21:
        recommendations.append({
            "factor": "Weekly gaming sessions",
            "current": f"{weekly_sessions} sessions/week",
            "advice": (
                "You report frequent gaming sessions. Consider reducing "
                "the number of sessions and setting gaming-free periods "
                "during the week."
            )
        })

    if night_gaming_ratio >= 0.50:
        recommendations.append({
            "factor": "Night gaming ratio",
            "current": f"{night_gaming_ratio:.0%}",
            "advice": (
                "A large proportion of your gaming takes place at night. "
                "Move gaming sessions earlier and set a cut-off time "
                "before bedtime."
            )
        })

    if toxic_exposure >= 0.50:
        recommendations.append({
            "factor": "Toxic gaming exposure",
            "current": f"{toxic_exposure:.0%}",
            "advice": (
                "Your reported exposure to toxic gaming interactions "
                "is high. Use mute/block/report tools, avoid hostile "
                "sessions and choose more supportive gaming communities."
            )
        })

    # Only mention sleep or exercise when the user's own input suggests
    # that the factor may need attention. Do not advise changes when okay.
    if sleep_hours < 7:
        recommendations.append({
            "factor": "Sleep duration",
            "current": f"{sleep_hours:.1f} hours/day",
            "advice": (
                "Your reported sleep duration is below the 7-hour "
                "review threshold. Consider protecting a consistent "
                "sleep schedule and avoiding gaming close to bedtime."
            )
        })

    if exercise_hours < 1:
        recommendations.append({
            "factor": "Exercise duration",
            "current": f"{exercise_hours:.1f} hours/day",
            "advice": (
                "Your reported exercise time is low. Consider adding "
                "short walks or other physical activities between "
                "gaming sessions."
            )
        })

    if screen_time_total >= 12:
        recommendations.append({
            "factor": "Total screen time",
            "current": f"{screen_time_total:.1f} hours/day",
            "advice": (
                "Your total daily screen time is high. Review non-essential "
                "screen activities and schedule regular screen-free breaks."
            )
        })

    return recommendations

# ---------------------------------------------------------
# Light / Dark mode and original GamingInsight styling
# ---------------------------------------------------------

CSS_PATH = Path(__file__).resolve().parent / "styles" / "style.css"
if CSS_PATH.exists():
    base_css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
else:
    st.warning(f"Website stylesheet not found: {CSS_PATH}")

# ---------------------------------------------------------
# Horizontal website navigation (same browser tab)
# ---------------------------------------------------------

# Read the selected page from the URL. Navigation is handled by
# Streamlit buttons below, not HTML links/window.open().
VALID_PAGES = [
    "Overview",
    "Gaming Behaviour",
    "Mental Health Indicators",
    "Addiction Prediction",
    "Prediction History",
    "Stakeholder Insights",
    "About Project",
]

page = st.query_params.get("page", "Overview")
if page not in VALID_PAGES:
    page = "Overview"

# Fixed dark appearance: the theme switcher has been removed.
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"

st.markdown(
    """
    <div class="gi-topbar gi-topbar-brand">
      <div class="gi-brand">
        <span class="gi-logo-icon">🎮</span>
        <span>Gaming<span class="gradient-text">Insights</span></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Streamlit buttons navigate in the current browser tab.
# The selected page is stored in the URL query parameter.
NAV_ITEMS = [
    ("Home / Overview", "Overview"),
    ("Gaming Behaviour", "Gaming Behaviour"),
    ("Mental Health Indicators", "Mental Health Indicators"),
    ("Addiction Prediction", "Addiction Prediction"),
    ("Prediction History", "Prediction History"),
    ("Stakeholder Insights", "Stakeholder Insights"),
    ("About Project", "About Project"),
]

# CSS styles the native Streamlit buttons to resemble the original navbar.
st.markdown(
    """
    <style>
    .st-key-gi_navbar div[data-testid="stButton"] button {
        width: 100%;
        min-height: 42px;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: var(--muted) !important;
        font-size: 0.88rem !important;
        font-weight: 650 !important;
        padding: 0.55rem 0.35rem !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        border-bottom: 2px solid transparent !important;
    }
    .st-key-gi_navbar div[data-testid="stButton"] button:hover {
        color: var(--text) !important;
        background: rgba(124, 92, 255, 0.10) !important;
        border-bottom-color: #8b6cff !important;
    }
    .st-key-gi_navbar div[data-testid="stButton"] button[kind="primary"] {
        color: var(--text) !important;
        background: transparent !important;
        border-bottom: 2px solid #8b6cff !important;
    }
    .st-key-gi_navbar div[data-testid="stButton"] button:focus {
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="gi_navbar"):
    nav_columns = st.columns([0.95, 1.05, 1.38, 1.20, 1.12, 1.25, 0.92], gap="small")

    for column, (label, target_page) in zip(nav_columns, NAV_ITEMS):
        with column:
            is_active = page == target_page
            if st.button(
                label,
                key=f"nav_{target_page.lower().replace(' ', '_').replace('/', '_')}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.query_params["page"] = target_page
                st.rerun()

# Compact project status line
st.markdown(
    f'<div class="gi-status-line"><span>Dataset records: <b>{len(df):,}</b></span>'
    f'<span>Saved prediction records: <b>{len(live_predictions_df):,}</b></span>'
    f'<span>Model: <b>{model_name}</b></span></div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------

if page == "Overview":

    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">● GAMING ANALYTICS &amp; INSIGHTS PLATFORM</div>
              <h1>Understand Your<br><span class="gradient-text">Gaming Behaviour.</span></h1>
              <p>Explore gaming patterns, examine selected mental health indicators, and understand potential gaming addiction risks through big data analytics, interactive visualisation and machine learning.</p>
              <div class="actions">
                <a class="btn btn-primary" href="?page=Gaming%20Behaviour">Explore Analytics →</a>
                <a class="btn btn-secondary" href="?page=About%20Project">About the Project</a>
              </div>
              <p class="hero-tagline">Data-driven insights. Better gaming awareness.</p>
            </div>
            <div class="panel">
              <div class="panel-head"><h3>Analytics at a Glance</h3><span class="pill">Live platform</span></div>
              <div class="stats-grid gi-hero-stats">
                <div class="stat-card"><div class="stat-icon">📁</div><h3>{len(df):,}</h3><p>Dataset records</p></div>
                <div class="stat-card"><div class="stat-icon">🎮</div><h3>03</h3><p>Core analytical themes</p></div>
                <div class="stat-card"><div class="stat-icon">⚡</div><h3>Big Data</h3><p>Processing approach</p></div>
                <div class="stat-card"><div class="stat-icon">📊</div><h3>Live</h3><p>Interactive visualisation</p></div>
              </div>
              <div class="gi-mini-chart">
                <h3>Gaming Behaviour Analysis</h3>
                <div class="mini-bars">
                  <span style="height:38%"></span><span style="height:58%"></span>
                  <span style="height:45%"></span><span style="height:76%"></span>
                  <span style="height:52%"></span><span style="height:88%"></span>
                  <span style="height:66%"></span>
                </div>
                <p class="note">Illustrative visual preview · Actual dataset analysis below</p>
              </div>
            </div>
          </div>
        </section>
        <div class="section-head gi-platform-head">
          <div class="section-label">PLATFORM AT A GLANCE</div>
          <h2 class="section-title">Your Gaming Data, <span class="gradient-text">Decoded.</span></h2>
          <p class="lead">Explore data-driven insights across gaming behaviour, mental health indicators and addiction prediction.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df) + len(live_predictions_df):,}"
    )

    col2.metric(
        "Average Gaming Hours",
        f"{df['daily_gaming_hours'].mean():.2f}"
    )

    col3.metric(
        "Average Addiction Level",
        f"{df['addiction_level'].mean():.2f}"
    )

    col4.metric(
        "Average Screen Time",
        f"{df['screen_time_total'].mean():.2f}"
    )

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        category_counts = (
            df["addiction_category"]
            .value_counts()
            .reindex(
                ["Low", "Moderate", "High"]
            )
            .reset_index()
        )

        category_counts.columns = [
            "Addiction Category",
            "Number of Records"
        ]

        category_figure = px.pie(
            category_counts,
            names="Addiction Category",
            values="Number of Records",
            hole=0.55,
            title="Distribution of Addiction Categories",
            color="Addiction Category",
            color_discrete_map={
                "Low": "#2ECC71",
                "Moderate": "#F1C40F",
                "High": "#E74C3C"
            }
        )

        st.plotly_chart(
            category_figure,
            use_container_width=True
        )

    with chart_col2:

        addiction_figure = px.histogram(
            df_chart,
            x="addiction_level",
            nbins=40,
            title="Distribution of Addiction Scores",
            labels={
                "addiction_level":
                    "Addiction Level"
            },
            color_discrete_sequence=[
                "#3366CC"
            ]
        )

        addiction_figure.update_layout(
            yaxis_title="Number of Records",
            showlegend=False
        )

        st.plotly_chart(
            addiction_figure,
            use_container_width=True
        )

    st.info(
        "The dashboard charts use a reproducible sample "
        f"of {chart_sample_size:,} records where necessary "
        "to maintain interactive performance."
    )


# ---------------------------------------------------------
# Page 2: Gaming Behaviour
# ---------------------------------------------------------

elif page == "Gaming Behaviour":

    st.markdown(
        '<p class="main-title">'
        'Gaming Behaviour Analysis'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Explore distributions and relationships among '
        'the main gaming behaviour variables.'
        '</p>',
        unsafe_allow_html=True
    )

    if not live_predictions_df.empty:
        st.info(
            f"Live prediction overlay: {len(live_predictions_df):,} "
            "saved prediction inputs are included in these gaming "
            "behaviour metrics and charts. Historical Parquet data "
            "remains unchanged."
        )
    else:
        st.caption(
            "No saved live prediction inputs are available yet. "
            "Charts currently reflect the historical dataset."
        )

    gaming_variables = [
        "daily_gaming_hours",
        "weekly_sessions",
        "years_gaming",
        "night_gaming_ratio",
        "weekend_gaming_hours",
        "multiplayer_ratio",
        "toxic_exposure",
        "violent_games_ratio",
        "mobile_gaming_ratio",
        "streaming_hours",
        "microtransactions_spending",
        "screen_time_total"
    ]

    selected_variable = st.selectbox(
        "Select a gaming variable",
        gaming_variables,
        format_func=format_variable_name
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Mean",
        f"{df_gaming_combined[selected_variable].mean():.2f}"
    )

    metric2.metric(
        "Median",
        f"{df_gaming_combined[selected_variable].median():.2f}"
    )

    metric3.metric(
        "Minimum",
        f"{df_gaming_combined[selected_variable].min():.2f}"
    )

    metric4.metric(
        "Maximum",
        f"{df_gaming_combined[selected_variable].max():.2f}"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        distribution_figure = px.histogram(
            df_chart_combined,
            x=selected_variable,
            nbins=40,
            title=(
                f"Distribution of "
                f"{format_variable_name(selected_variable)}"
            ),
            color_discrete_sequence=["#3366CC"]
        )

        st.plotly_chart(
            distribution_figure,
            use_container_width=True
        )

    with chart_col2:

        box_figure = px.box(
            df_chart_combined,
            x="addiction_category",
            y=selected_variable,
            category_orders={
                "addiction_category": [
                    "Low",
                    "Moderate",
                    "High"
                ]
            },
            color="addiction_category",
            color_discrete_map={
                "Low": "#2ECC71",
                "Moderate": "#F1C40F",
                "High": "#E74C3C"
            },
            title=(
                f"{format_variable_name(selected_variable)} "
                "by Addiction Category"
            )
        )

        st.plotly_chart(
            box_figure,
            use_container_width=True
        )

    scatter_figure = px.scatter(
        df_chart_combined.sample(
            min(10_000, len(df_chart_combined)),
            random_state=42
        ),
        x=selected_variable,
        y="addiction_level",
        opacity=0.35,
        trendline="ols",
        title=(
            f"{format_variable_name(selected_variable)} "
            "and Addiction Level"
        ),
        labels={
            "addiction_level":
                "Addiction Level"
        }
    )

    st.plotly_chart(
        scatter_figure,
        use_container_width=True
    )


# ---------------------------------------------------------
# Page 3: Mental Health Indicators
# ---------------------------------------------------------

elif page == "Mental Health Indicators":

    st.markdown(
        '<p class="main-title">'
        'Mental Health Indicator Analysis'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Descriptive exploration of mental health '
        'indicators in relation to gaming addiction.'
        '</p>',
        unsafe_allow_html=True
    )

    mental_health_variables = [
        "stress_level",
        "anxiety_score",
        "depression_score",
        "loneliness_score",
        "happiness_score"
    ]

    selected_indicator = st.selectbox(
        "Select a mental health indicator",
        mental_health_variables,
        format_func=format_variable_name
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean",
        f"{df[selected_indicator].mean():.2f}"
    )

    col2.metric(
        "Median",
        f"{df[selected_indicator].median():.2f}"
    )

    col3.metric(
        "Standard Deviation",
        f"{df[selected_indicator].std():.2f}"
    )

    mental_col1, mental_col2 = st.columns(2)

    with mental_col1:

        mental_distribution = px.histogram(
            df_chart,
            x=selected_indicator,
            nbins=40,
            title=(
                f"Distribution of "
                f"{format_variable_name(selected_indicator)}"
            ),
            color_discrete_sequence=["#8E44AD"]
        )

        st.plotly_chart(
            mental_distribution,
            use_container_width=True
        )

    with mental_col2:

        mental_box = px.box(
            df_chart,
            x="addiction_category",
            y=selected_indicator,
            color="addiction_category",
            category_orders={
                "addiction_category": [
                    "Low",
                    "Moderate",
                    "High"
                ]
            },
            color_discrete_map={
                "Low": "#2ECC71",
                "Moderate": "#F1C40F",
                "High": "#E74C3C"
            },
            title=(
                f"{format_variable_name(selected_indicator)} "
                "by Addiction Category"
            )
        )

        st.plotly_chart(
            mental_box,
            use_container_width=True
        )

    correlation_columns = (
        mental_health_variables
        + [
            "daily_gaming_hours",
            "screen_time_total",
            "addiction_level"
        ]
    )

    mental_correlation = (
        df[correlation_columns]
        .corr()
        .round(2)
    )

    correlation_figure = px.imshow(
        mental_correlation,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=(
            "Correlation of Gaming, Addiction and "
            "Mental Health Indicators"
        )
    )

    st.plotly_chart(
        correlation_figure,
        use_container_width=True
    )

    st.warning(
        "These values are analysed as dataset indicators. "
        "They do not provide a clinical mental health "
        "assessment, and correlation does not prove causation."
    )


# ---------------------------------------------------------
# Page 4: Prediction
# ---------------------------------------------------------

elif page == "Addiction Prediction":

    st.markdown(
        '<p class="main-title">'
        'Gaming Addiction Level Prediction'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Enter gaming behaviour and lifestyle information '
        'to generate an analytical prediction.'
        '</p>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        input_col1, input_col2 = st.columns(2)

        with input_col1:

            daily_gaming_hours = st.number_input(
                "Daily gaming hours",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5
            )

            weekly_sessions = st.number_input(
                "Weekly gaming sessions",
                min_value=0,
                max_value=50,
                value=20,
                step=1
            )

            night_gaming_ratio = st.slider(
                "Night gaming ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

            toxic_exposure = st.slider(
                "Toxic gaming exposure",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01
            )

        with input_col2:

            sleep_hours = st.number_input(
                "Daily sleep hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

            exercise_hours = st.number_input(
                "Daily exercise hours",
                min_value=0.0,
                max_value=24.0,
                value=1.5,
                step=0.5
            )

            screen_time_total = st.number_input(
                "Total daily screen time",
                min_value=0.0,
                max_value=24.0,
                value=8.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        if screen_time_total < daily_gaming_hours:

            st.error(
                "Total screen time cannot be lower than "
                "daily gaming hours. Please correct the values."
            )

        else:

            input_data = pd.DataFrame({
                "daily_gaming_hours": [
                    daily_gaming_hours
                ],
                "weekly_sessions": [
                    weekly_sessions
                ],
                "night_gaming_ratio": [
                    night_gaming_ratio
                ],
                "toxic_exposure": [
                    toxic_exposure
                ],
                "sleep_hours": [
                    sleep_hours
                ],
                "exercise_hours": [
                    exercise_hours
                ],
                "screen_time_total": [
                    screen_time_total
                ]
            })

            input_data = input_data[
                model_features
            ]

            prediction = model.predict(
                input_data
            )[0]

            prediction = float(
                np.clip(
                    prediction,
                    0,
                    10
                )
            )

            category = convert_addiction_category(
                prediction
            )

            # Save the generated prediction to Supabase.
            if supabase_connected and supabase_client is not None:
                prediction_record = {
                    "input_features": {
                        "daily_gaming_hours": float(daily_gaming_hours),
                        "weekly_sessions": int(weekly_sessions),
                        "night_gaming_ratio": float(night_gaming_ratio),
                        "toxic_exposure": float(toxic_exposure),
                        "sleep_hours": float(sleep_hours),
                        "exercise_hours": float(exercise_hours),
                        "screen_time_total": float(screen_time_total)
                    },
                    "predicted_score": float(prediction),
                    "addiction_category": category,
                    "model_name": model_name,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }

                try:
                    (
                        supabase_client
                        .table(PREDICTION_TABLE)
                        .insert(prediction_record)
                        .execute()
                    )
                    st.success("Prediction successfully saved to Supabase.")
                except Exception as error:
                    st.warning(
                        f"Prediction generated, but could not be saved to Supabase: {error}"
                    )
            else:
                st.warning(
                    "Supabase is unavailable. This prediction was not saved."
                )

            result_colour = category_colour(
                category
            )

            result_col1, result_col2 = st.columns(
                [1.4, 1]
            )

            with result_col1:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        number={
                            "suffix": " / 10",
                            "valueformat": ".2f"
                        },
                        title={
                            "text":
                                "Predicted Addiction Level"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 10]
                            },
                            "bar": {
                                "color": result_colour
                            },
                            "steps": [
                                {
                                    "range": [0, 3],
                                    "color": "#D5F5E3"
                                },
                                {
                                    "range": [3, 6],
                                    "color": "#FCF3CF"
                                },
                                {
                                    "range": [6, 10],
                                    "color": "#FADBD8"
                                }
                            ],
                            "threshold": {
                                "line": {
                                    "color": result_colour,
                                    "width": 5
                                },
                                "value": prediction
                            }
                        }
                    )
                )

                gauge.update_layout(
                    height=350,
                    margin=dict(
                        l=30,
                        r=30,
                        t=70,
                        b=20
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            with result_col2:

                st.markdown("### Prediction Outcome")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2 style="color:{result_colour};">
                            {category}
                        </h2>
                        <p>
                            Predicted addiction score:
                            <strong>{prediction:.2f}/10</strong>
                        </p>
                        <p>
                            Prediction model:
                            <strong>{model_name}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if category == "Low":

                st.success(
                    "The entered behaviour is associated "
                    "with a low analytical addiction level."
                )

            elif category == "Moderate":

                st.warning(
                    "The entered behaviour is associated "
                    "with a moderate analytical addiction "
                    "level. Reviewing gaming and screen-time "
                    "habits may be beneficial."
                )

            else:

                st.error(
                    "The entered behaviour is associated "
                    "with a high analytical addiction level. "
                    "Consider reviewing gaming habits and "
                    "seeking appropriate support if gaming "
                    "is affecting daily life."
                )

            # ---------------------------------------------------------
            # Prescriptive Analytics: personalised recommendations
            # ---------------------------------------------------------
            st.markdown("### Personalised Recommendations")
            st.caption(
                "Recommendations are generated from the submitted input. "
                "Only factors that cross the configured review thresholds "
                "are highlighted; factors within range are not flagged."
            )

            recommendations = generate_prescriptive_recommendations(
                daily_gaming_hours=daily_gaming_hours,
                weekly_sessions=weekly_sessions,
                night_gaming_ratio=night_gaming_ratio,
                toxic_exposure=toxic_exposure,
                sleep_hours=sleep_hours,
                exercise_hours=exercise_hours,
                screen_time_total=screen_time_total
            )

            if recommendations:
                for item in recommendations:
                    with st.container(border=True):
                        st.markdown(f"**{item['factor']}**")
                        st.caption(f"Your input: {item['current']}")
                        st.write(item["advice"])
            else:
                st.success(
                    "No specific behaviour factor crossed the current "
                    "recommendation thresholds. Continue maintaining "
                    "your existing habits and review them regularly."
                )

            st.caption(
                "Note: These are rule-based educational suggestions, "
                "not medical advice. Thresholds and targets should be "
                "validated against the project dataset and supervisor "
                "guidance; they do not guarantee a lower predicted score."
            )

            st.markdown(
                """
                <div class="disclaimer">
                    <strong>Important:</strong>
                    This result is generated from patterns in
                    the project dataset. It is not a medical
                    diagnosis and should not replace assessment
                    by a qualified professional.
                </div>
                """,
                unsafe_allow_html=True
            )



# ---------------------------------------------------------
# Page 5: Prediction History
# ---------------------------------------------------------

elif page == "Prediction History":

    st.markdown(
        '<p class="main-title">Prediction History</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Review predictions previously generated and saved in Supabase.'
        '</p>',
        unsafe_allow_html=True
    )

    if not supabase_connected or supabase_client is None:
        st.warning(
            "Supabase is currently unavailable. Prediction history cannot be loaded."
        )
    else:
        try:
            history_response = (
                supabase_client
                .table(PREDICTION_TABLE)
                .select("*")
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )
            history_documents = history_response.data or []

            if not history_documents:
                st.info(
                    "No prediction records found yet. Generate a prediction "
                    "to create the first history record."
                )
            else:
                history_df = pd.DataFrame(history_documents)
                total_predictions = len(history_df)
                average_score = history_df["predicted_score"].mean()
                latest_category = history_df.iloc[0]["addiction_category"]

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Records shown", f"{total_predictions:,}")
                metric_col2.metric("Average predicted score", f"{average_score:.2f}/10")
                metric_col3.metric("Latest category", latest_category)

                st.divider()

                category_counts = (
                    history_df["addiction_category"]
                    .value_counts()
                    .reindex(["Low", "Moderate", "High"], fill_value=0)
                    .rename_axis("Addiction Category")
                    .reset_index(name="Predictions")
                )

                history_col1, history_col2 = st.columns(2)
                with history_col1:
                    history_chart = px.bar(
                        category_counts,
                        x="Addiction Category",
                        y="Predictions",
                        color="Addiction Category",
                        color_discrete_map={
                            "Low": "#2ECC71",
                            "Moderate": "#F1C40F",
                            "High": "#E74C3C"
                        },
                        title="Prediction Categories"
                    )
                    st.plotly_chart(history_chart, use_container_width=True)

                with history_col2:
                    score_chart = px.histogram(
                        history_df,
                        x="predicted_score",
                        nbins=20,
                        title="Distribution of Predicted Scores",
                        labels={"predicted_score": "Predicted score"}
                    )
                    st.plotly_chart(score_chart, use_container_width=True)

                display_df = history_df.copy()
                if "created_at" in display_df.columns:
                    display_df["created_at"] = pd.to_datetime(
                        display_df["created_at"], utc=True, errors="coerce"
                    ).dt.strftime("%Y-%m-%d %H:%M:%S UTC")

                if "input_features" in display_df.columns:
                    display_df["input_features"] = display_df[
                        "input_features"
                    ].apply(
                        lambda values: ", ".join(
                            f"{key}: {value}" for key, value in values.items()
                        ) if isinstance(values, dict) else str(values)
                    )

                st.markdown("### Saved Prediction Records")
                st.dataframe(
                    display_df[
                        [
                            column for column in [
                                "created_at",
                                "predicted_score",
                                "addiction_category",
                                "model_name",
                                "input_features"
                            ]
                            if column in display_df.columns
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                if st.button("Refresh Prediction History"):
                    st.rerun()

        except Exception as error:
            st.error(f"Unable to retrieve prediction history from Supabase: {error}")



# ---------------------------------------------------------
# Page: Stakeholder Insights
# ---------------------------------------------------------

elif page == "Stakeholder Insights":

    st.markdown(
        '<p class="main-title">Stakeholder Insights</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Explore how gaming behaviour analytics can inform gamers, parents, public health stakeholders and the research community.</p>',
        unsafe_allow_html=True
    )

    # Use the project dataset directly; no illustrative or fabricated records.
    stakeholder_df = df.copy()
    numeric_columns = stakeholder_df.select_dtypes(include=np.number).columns.tolist()

    if "addiction_level" not in stakeholder_df.columns:
        st.warning("The dataset does not contain the addiction_level field required for this page.")
    else:
        stakeholder_df["addiction_category"] = stakeholder_df["addiction_level"].apply(
            convert_addiction_category
        )

        total_records = len(stakeholder_df)
        avg_gaming = stakeholder_df["daily_gaming_hours"].mean() if "daily_gaming_hours" in stakeholder_df else np.nan
        avg_addiction = stakeholder_df["addiction_level"].mean()
        high_share = (stakeholder_df["addiction_category"] == "High").mean() * 100

        st.markdown("### Key findings from the dataset")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Dataset records", f"{total_records:,}")
        metric_cols[1].metric("Average addiction score", f"{avg_addiction:.2f} / 10")
        metric_cols[2].metric("High category", f"{high_share:.1f}%")
        metric_cols[3].metric(
            "Average daily gaming",
            f"{avg_gaming:.2f} hrs" if pd.notna(avg_gaming) else "N/A"
        )
        st.caption("Metrics are calculated from the project dataset currently loaded by the application. Addiction categories use the project's score bands: Low (≤3), Moderate (>3 to 6), and High (>6).")

        st.markdown("### Analytical evidence")
        chart_left, chart_right = st.columns(2)

        with chart_left:
            if {"daily_gaming_hours", "addiction_level"}.issubset(stakeholder_df.columns):
                corr_value = stakeholder_df[["daily_gaming_hours", "addiction_level"]].corr().iloc[0, 1]
                scatter_df = stakeholder_df[["daily_gaming_hours", "addiction_level"]].dropna()
                if len(scatter_df) > 5000:
                    scatter_df = scatter_df.sample(5000, random_state=42)
                fig_scatter = px.scatter(
                    scatter_df,
                    x="daily_gaming_hours",
                    y="addiction_level",
                    opacity=0.35,
                    labels={"daily_gaming_hours": "Daily gaming hours", "addiction_level": "Addiction level score"},
                    title=f"Daily Gaming Hours and Addiction Level (r = {corr_value:.2f})"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Daily gaming hours data is unavailable for this visualisation.")

        with chart_right:
            correlation_features = [
                col for col in [
                    "daily_gaming_hours", "weekly_sessions", "night_gaming_ratio",
                    "toxic_exposure", "sleep_hours", "exercise_hours",
                    "screen_time_total", "addiction_level"
                ] if col in numeric_columns
            ]
            if len(correlation_features) >= 2:
                corr_target = (
                    stakeholder_df[correlation_features]
                    .corr(numeric_only=True)["addiction_level"]
                    .drop("addiction_level", errors="ignore")
                    .sort_values()
                    .reset_index()
                )
                corr_target.columns = ["Attribute", "Correlation"]
                corr_target["Attribute"] = corr_target["Attribute"].map(format_variable_name)
                fig_corr = px.bar(
                    corr_target,
                    x="Correlation",
                    y="Attribute",
                    orientation="h",
                    title="Correlation with Addiction Level",
                    range_x=[-1, 1],
                    color="Correlation",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("There are not enough selected numerical fields to calculate correlations.")

        if {"stress_level", "anxiety_score", "happiness_score"}.intersection(stakeholder_df.columns):
            st.markdown("### Mental health indicators across addiction categories")
            indicator_columns = [
                col for col in ["stress_level", "anxiety_score", "happiness_score"]
                if col in stakeholder_df.columns and pd.api.types.is_numeric_dtype(stakeholder_df[col])
            ]
            if indicator_columns:
                indicator_summary = (
                    stakeholder_df.groupby("addiction_category")[indicator_columns]
                    .mean()
                    .reindex(["Low", "Moderate", "High"])
                    .reset_index()
                    .melt(id_vars="addiction_category", var_name="Indicator", value_name="Average")
                )
                indicator_summary["Indicator"] = indicator_summary["Indicator"].map(format_variable_name)
                fig_indicators = px.bar(
                    indicator_summary,
                    x="addiction_category",
                    y="Average",
                    color="Indicator",
                    barmode="group",
                    category_orders={"addiction_category": ["Low", "Moderate", "High"]},
                    title="Average Selected Indicators by Addiction Category",
                    labels={"addiction_category": "Addiction category", "Average": "Average value"}
                )
                st.plotly_chart(fig_indicators, use_container_width=True)

        st.markdown("### Insights for different stakeholders")
        gamer_tab, parent_tab, health_tab, research_tab = st.tabs([
            "🎮 Gamers", "👨‍👩‍👧 Parents", "🏥 Public Health", "🔬 Researchers"
        ])

        with gamer_tab:
            st.markdown("**Understanding personal gaming habits**")
            st.write("Gamers can use the dashboard findings to reflect on daily gaming duration, weekly sessions, night gaming and overall screen time. The addiction prediction page provides an estimated score based on the selected gaming and lifestyle inputs.")
            st.info("Suggested use: review gaming patterns regularly, set personal time boundaries and balance gaming with sleep, exercise, study, work and social activities.")

        with parent_tab:
            st.markdown("**Supporting healthy digital boundaries**")
            st.write("Parents can explore dataset-level patterns to better understand gaming habits that may warrant discussion, such as extended daily gaming and high screen time. The charts are intended to support awareness and communication, not to label or diagnose a child.")
            st.info("Suggested use: discuss reasonable gaming schedules, encourage breaks and maintain a balanced routine that includes sleep, physical activity and offline interests.")

        with health_tab:
            st.markdown("**Informing awareness and further investigation**")
            st.write("Public health stakeholders may use the aggregated findings as supplementary information when considering problematic gaming behaviour and selected mental health indicators. The dataset can help identify patterns for awareness activities and further research.")
            st.info("Limitation: these dataset associations do not establish causation and should not be treated as clinical evidence or population-wide prevalence estimates.")

        with research_tab:
            st.markdown("**Demonstrating a Big Data analytics workflow**")
            st.write("The project demonstrates the use of PySpark for data preparation and large-scale processing, a trained regression model for estimating addiction levels, and Streamlit with interactive visualisations for communicating analytical results.")
            st.info("Researchers can use the presented workflow and findings as a reference point for further investigation, while considering dataset quality, feature limitations and model evaluation results.")

        st.markdown("### Practical recommendations")
        recommendation_cols = st.columns(3)
        with recommendation_cols[0]:
            st.markdown("**Balanced gaming routine**")
            st.write("Encourage planned gaming periods, regular breaks and a balance with daily responsibilities.")
        with recommendation_cols[1]:
            st.markdown("**Healthy digital habits**")
            st.write("Pay attention to late-night gaming, sleep routines, screen time and exposure to toxic interactions.")
        with recommendation_cols[2]:
            st.markdown("**Evidence-aware decisions**")
            st.write("Use dashboard findings as supplementary insights and avoid interpreting predictions as medical diagnoses.")

        st.caption("Methodological note: correlation indicates association rather than causation. Predicted addiction scores are model estimates for educational and analytical purposes, not clinical diagnoses.")

# ---------------------------------------------------------
# Page 7: About
# ---------------------------------------------------------

elif page == "About Project":

    st.markdown(
        '<p class="main-title">About the Project</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Project Title

        **Big Data Analytics and Visualization of Gaming
        Behaviours and Impact on Mental Health**

        ### Project Purpose

        This project analyses a large gaming dataset to
        identify patterns in gaming behaviour, addiction
        levels, lifestyle factors, and mental health
        indicators.

        The system combines:

        - Big-data-oriented data processing
        - Exploratory data analysis
        - Interactive data visualisation
        - Machine-learning regression
        - User-friendly addiction-level prediction

        ### Prediction Scope

        The predictive model estimates an addiction score
        from 0 to 10 using gaming behaviour and lifestyle
        variables. Age and mental health indicators are not
        used as prediction inputs.

        ### Ethical Limitation

        All dashboard results are educational analytical
        indicators. They must not be interpreted as clinical
        diagnoses or proof of cause-and-effect relationships.
        """
    )