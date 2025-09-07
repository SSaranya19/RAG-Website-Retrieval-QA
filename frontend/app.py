import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chat — Q&A with Citations", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.title("🔎 RAG Chat — Q&A with Citations")

# ---------------- AUTH ----------------
if "token" not in st.session_state:
    st.session_state.token = None

if not st.session_state.token:
    with st.form("login"):
        st.subheader("🔐 Please log in")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            try:
                resp = requests.post(
                    f"{API_URL}/api/v1/login",
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                resp.raise_for_status()
                st.session_state.token = resp.json()["access_token"]
                st.success("✅ Login successful!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Login failed: {e}")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------- SIDEBAR: Index websites ----------------
st.sidebar.subheader("🌐 Index websites")

# Show currently indexed URLs
try:
    resp = requests.get(f"{API_URL}/api/v1/index", headers=headers)
    resp.raise_for_status()
    indexed_urls = resp.json().get("indexed", [])
    if indexed_urls:
        st.sidebar.markdown("**Currently Indexed URLs:**")
        for url in indexed_urls:
            st.sidebar.write(f"🔗 {url}")
    else:
        st.sidebar.info("No URLs indexed yet.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not fetch indexed URLs: {e}")
    indexed_urls = []

# Input box to add new URLs
urls_input = st.sidebar.text_area("Enter URLs (one per line)")
if st.sidebar.button("🚀 Index URLs"):
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/index",
            headers=headers,
            json={"urls": urls_input.splitlines(), "mode": "replace"}
        )
        resp.raise_for_status()
        st.sidebar.success("✅ URLs indexed successfully")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Index failed: {e}")

# ---------------- MAIN CHAT ----------------
user_q = st.text_input("💬 Ask a question...")

if st.button("Send") and user_q.strip():
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/chat?top_k=5",
            headers=headers,
            json={"messages": [{"role": "user", "content": user_q}]}
        )
        resp.raise_for_status()
        data = resp.json()

        st.markdown("### Answer:")
        st.markdown(data["answer"]["content"], unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
