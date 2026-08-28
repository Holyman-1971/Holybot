import os
import json
import pandas as pd
from datetime import datetime
import streamlit as st

st.set_page_config(
    page_title="Holybot 2.0 | Log Inspector",
    page_icon="📜",
    layout="wide"
)

LOG_FILE = "./logs/chat_logs.jsonl"

st.title("📜 Holybot 2.0 Log Inspector")
st.caption("Local JSONL Session Parser & Analytics")

# -----------------------------------------------------------------------------
# 1. Data Loader
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)  # Auto-refresh cache every 5 seconds
def load_logs():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    # Format timestamps
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["formatted_time"] = df["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["formatted_time"] = "Unknown"
        
    return df

df = load_logs()

if df.empty:
    st.warning(f"⚠️ No log entries found at `{LOG_FILE}`. Interact with `app.py` with logging enabled to generate entries.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Sidebar Filters & Metrics
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Log Filters")

# Metric counters
st.sidebar.metric("Total Logs", len(df))
st.sidebar.metric("Unique Sessions", df["session_id"].nunique() if "session_id" in df.columns else 0)

# Filter by Session ID
session_options = ["All"] + list(df["session_id"].unique()) if "session_id" in df.columns else ["All"]
selected_session = st.sidebar.selectbox("Filter by Session ID", session_options)

# Filter by Mode
mode_options = ["All"] + list(df["mode"].unique()) if "mode" in df.columns else ["All"]
selected_mode = st.sidebar.selectbox("Filter by Mode", mode_options)

# Keyword search
search_query = st.sidebar.text_input("Search Prompts or Responses", "").strip().lower()

# Apply Filters
filtered_df = df.copy()

if selected_session != "All":
    filtered_df = filtered_df[filtered_df["session_id"] == selected_session]

if selected_mode != "All":
    filtered_df = filtered_df[filtered_df["mode"] == selected_mode]

if search_query:
    filtered_df = filtered_df[
        filtered_df["user_prompt"].str.lower().str.contains(search_query, na=False) |
        filtered_df["bot_response"].str.lower().str.contains(search_query, na=False)
    ]

# Sort newest first
filtered_df = filtered_df.sort_index(ascending=False)

st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** log entries")

# -----------------------------------------------------------------------------
# 3. Main GUI Log Feed
# -----------------------------------------------------------------------------
st.markdown("---")

for idx, row in filtered_df.iterrows():
    # Sticky header info using expander title & columns
    time_str = row.get("formatted_time", "N/A")
    mode_str = row.get("mode", "Standard")
    session_str = row.get("session_id", "Unknown")
    prompt_snippet = row.get("user_prompt", "")
    
    # Truncate title snippet if prompt is long
    title_display = f"[{time_str}] | Mode: {mode_str} | Prompt: {prompt_snippet[:60]}..."
    
    with st.expander(title_display, expanded=True):
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**Timestamp:** `{time_str}`")
        with col2:
            st.markdown(f"**Mode:** `{mode_str}`")
        with col3:
            st.markdown(f"**Session ID:** `{session_str}`")
            
        st.markdown("---")
        
        # Prominent display of user prompt and bot response
        st.markdown("**👤 User Prompt:**")
        st.info(row.get("user_prompt", ""))
        
        st.markdown("**🤖 Bot Response:**")
        st.success(row.get("bot_response", ""))

# -----------------------------------------------------------------------------
# 4. Export Utility
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Export Filtered Logs (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_chat_logs.csv",
    mime="text/csv"
)