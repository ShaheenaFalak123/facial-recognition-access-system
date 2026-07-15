"""Analytics dashboard for the access-control recognition_events data,
querying Supabase live on every load (not a static export).

Requires a SUPABASE_URL secret (Streamlit Cloud: Settings -> Secrets;
locally: the SUPABASE_URL environment variable) -- the connection
string is never committed to the repo.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Access Control Analytics", page_icon="📊", layout="wide")

# Fixed categorical colors, used consistently across every chart --
# never cycled per-chart. Blue/orange is a colorblind-safe pair
# (distinguishable under deuteranopia/protanopia).
COLOR_GENUINE = "#2563EB"
COLOR_IMPOSTOR = "#F97316"
COLOR_CRITICAL = "#DC2626"


def get_connection_string():
    # st.secrets raises (not just returns False) if no secrets.toml
    # exists at all, which is the normal case for local development --
    # so the secrets lookup itself needs to be guarded, not just the key.
    try:
        if "SUPABASE_URL" in st.secrets:
            return st.secrets["SUPABASE_URL"]
    except st.errors.StreamlitSecretNotFoundError:
        pass
    if os.environ.get("SUPABASE_URL"):
        return os.environ["SUPABASE_URL"]
    st.error("No SUPABASE_URL found in Streamlit secrets or environment variables.")
    st.stop()


@st.cache_resource
def get_conn():
    return create_engine(get_connection_string())


@st.cache_data(ttl=60)
def load_events():
    return pd.read_sql(
        "SELECT e.*, c.location, c.zone FROM recognition_events e "
        "JOIN cameras c ON c.camera_id = e.camera_id "
        "ORDER BY event_timestamp",
        get_conn(),
    )


@st.cache_data(ttl=60)
def load_anomalies():
    query = """
        WITH failed_attempts AS (
            SELECT event_id, camera_id, event_timestamp
            FROM recognition_events
            WHERE NOT is_match
        ),
        with_trailing_count AS (
            SELECT event_id, camera_id, event_timestamp,
                COUNT(*) OVER (
                    PARTITION BY camera_id ORDER BY event_timestamp
                    RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND CURRENT ROW
                ) AS failed_attempts_in_last_5min
            FROM failed_attempts
        )
        SELECT c.location, c.zone, w.event_timestamp, w.failed_attempts_in_last_5min
        FROM with_trailing_count w
        JOIN cameras c ON c.camera_id = w.camera_id
        WHERE w.failed_attempts_in_last_5min >= 3
        ORDER BY w.event_timestamp;
    """
    return pd.read_sql(query, get_conn())


st.title("Access Control Analytics")
st.caption(
    "Live queries against Supabase Postgres. Confidence scores come from the "
    "[face verification model](https://github.com/ShaheenaFalak123/facial-recognition-access-system)'s "
    "real predictions on its held-out test set; camera/timestamp metadata is synthesized."
)

events = load_events()
events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

days_back = st.slider("Show last N days", min_value=7, max_value=30, value=30)
cutoff = events["event_timestamp"].max() - pd.Timedelta(days=days_back)
filtered = events[events["event_timestamp"] >= cutoff]

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total attempts", len(filtered))
col2.metric("Overall match rate", f"{filtered['is_match'].mean():.1%}")
false_accepts = ((~filtered["is_genuine_attempt"]) & filtered["is_match"]).sum()
false_rejects = (filtered["is_genuine_attempt"] & ~filtered["is_match"]).sum()
col3.metric("False accepts", int(false_accepts))
col4.metric("False rejects", int(false_rejects))

st.divider()

# --- Daily volume: genuine vs impostor, stacked ---
daily = (
    filtered.assign(day=filtered["event_timestamp"].dt.date)
    .groupby(["day", "is_genuine_attempt"])
    .size()
    .unstack(fill_value=0)
)
fig = go.Figure()
fig.add_bar(
    x=daily.index, y=daily.get(True, 0), name="Genuine attempts", marker_color=COLOR_GENUINE
)
fig.add_bar(
    x=daily.index, y=daily.get(False, 0), name="Impostor attempts", marker_color=COLOR_IMPOSTOR
)
fig.update_layout(
    barmode="stack",
    title="Daily attempt volume",
    xaxis_title=None,
    yaxis_title="Attempts",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, width='stretch')

# --- Rolling match rate ---
rolling = filtered.sort_values("event_timestamp").copy()
rolling["rolling_match_rate"] = rolling["is_match"].rolling(10, min_periods=1).mean()
fig2 = go.Figure()
fig2.add_scatter(
    x=rolling["event_timestamp"],
    y=rolling["rolling_match_rate"],
    mode="lines",
    line=dict(color=COLOR_GENUINE, width=2),
    name="Rolling match rate (last 10 attempts)",
)
fig2.update_layout(title="Rolling match rate", yaxis_title=None, xaxis_title=None)
st.plotly_chart(fig2, width='stretch')

# --- Camera breakdown ---
by_camera = filtered.groupby("location").size().sort_values(ascending=False)
fig3 = go.Figure(go.Bar(x=by_camera.values, y=by_camera.index, orientation="h", marker_color=COLOR_GENUINE))
fig3.update_layout(title="Attempts by camera", xaxis_title="Attempts", yaxis_title=None)
st.plotly_chart(fig3, width='stretch')

st.divider()

# --- Anomaly detection ---
st.subheader("Flagged: 3+ failed attempts within 5 minutes at one camera")
anomalies = load_anomalies()
if anomalies.empty:
    st.write("No anomalies detected in the current data.")
else:
    st.dataframe(anomalies, width='stretch')
    st.caption(
        f"⚠️ {anomalies['location'].nunique()} camera(s) show a burst of failed attempts "
        "consistent with a spoofing/brute-force pattern."
    )
