import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# -----------------------------------------------------------------------------
# 1. Page Configuration & Environment
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Holybot 2.0 | Log Inspector",
    page_icon="📜",
    layout="wide"
)

LOG_FILE = "./logs/chat_logs.jsonl"

st.title("📜 Holybot 2.0 Log Inspector")
st.caption("Live Supabase Cloud & Local JSONL Session Parser")

supabase_url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", None)
supabase_key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", None)

# -----------------------------------------------------------------------------
# 2. Unified Data Loader (Supabase Cloud + Local Fallback)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)  # Auto-refresh data every 5 seconds
def load_logs():
    # Attempt 1: Fetch from Supabase Cloud DB
    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            response = supabase.table("chat_logs").select("*").order("created_at", desc=True).execute()
            data = response.data
            if data:
                df = pd.DataFrame(data)
                if "created_at" in df.columns:
                    df["formatted_time"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    df["formatted_time"] = "Unknown"
                df["source"] = "Supabase Cloud"
                return df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Supabase query failed, checking local logs: {e}")

    # Attempt 2: Local JSONL Fallback
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
    
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["formatted_time"] = df["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["formatted_time"] = "Unknown"
        
    df["source"] = "Local File"
    return df

df = load_logs()

if df.empty:
    st.warning("⚠️ No log entries found in Supabase Cloud or locally at `./logs/chat_logs.jsonl`.")
    st.info("Interact with `app.py` with logging enabled to generate initial log entries.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Metrics & Filters
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Log Filters")

source_label = df["source"].iloc[0] if "source" in df.columns else "Unknown"
st.sidebar.info(f"**Data Source:** `{source_label}`")

# Metric Counters
st.sidebar.metric("Total Logs", len(df))
st.sidebar.metric("Unique Sessions", df["session_id"].nunique() if "session_id" in df.columns else 0)

# Filter by Session ID
session_options = ["All"] + list(df["session_id"].unique()) if "session_id" in df.columns else ["All"]
selected_session = st.sidebar.selectbox("Filter by Session ID", session_options)

# Filter by Response Mode
mode_options = ["All"] + list(df["mode"].unique()) if "mode" in df.columns else ["All"]
selected_mode = st.sidebar.selectbox("Filter by Mode", mode_options)

# Keyword Search
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

# Ensure sorting (newest first)
if "id" in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by="id", ascending=False)
else:
    filtered_df = filtered_df.sort_index(ascending=False)

st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** log entries")

# -----------------------------------------------------------------------------
# 4. Main Log Inspector Interface
# -----------------------------------------------------------------------------
st.markdown("---")

for idx, row in filtered_df.iterrows():
    time_str = row.get("formatted_time", "N/A")
    mode_str = row.get("mode", "Standard")
    session_str = row.get("session_id", "Unknown")
    prompt_snippet = row.get("user_prompt", "")
    
    title_display = f"[{time_str}] | Mode: {mode_str} | Session: {session_str} | Prompt: {prompt_snippet[:50]}..."
    
    with st.expander(title_display, expanded=True):
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**Timestamp:** `{time_str}`")
        with col2:
            st.markdown(f"**Mode:** `{mode_str}`")
        with col3:
            st.markdown(f"**Session ID:** `{session_str}`")
            
        st.markdown("---")
        
        st.markdown("**👤 User Prompt:**")
        st.info(row.get("user_prompt", ""))
        
        st.markdown("**🤖 Bot Response:**")
        st.success(row.get("bot_response", ""))

# -----------------------------------------------------------------------------
# 5. CSV Export Utility
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Export Filtered Logs (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_chat_logs.csv",
    mime="text/csv"
)