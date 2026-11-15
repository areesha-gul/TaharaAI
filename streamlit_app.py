"""Streamlit interface for the evidence-grounded compliance assistant."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for key in st.secrets:
        value = st.secrets[key]
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))
except Exception:
    pass

from query import ask


st.set_page_config(page_title="TaharaAI", page_icon="T", layout="wide")

st.title("TaharaAI")
st.caption("Compliance research grounded in your indexed documents")

with st.sidebar:
    st.header("Search")
    retrieval_mode = st.radio(
        "Retrieval mode",
        options=["hybrid", "dense"],
        index=0,
        format_func=str.title,
        horizontal=True,
    )
    use_reranker = st.checkbox("Use reranker", value=False)
    st.caption("Reranking can improve ordering but adds model latency.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _source_title(source: dict) -> str:
    location = source.get("doc_name") or "Unknown document"
    page_nums = source.get("page_nums") or (
        [source["page_num"]] if source.get("page_num") else []
    )
    section_ids = source.get("section_ids") or (
        [source["section_id"]] if source.get("section_id") else []
    )
    details = []
    if page_nums:
        details.append(f"p. {', '.join(str(page) for page in page_nums)}")
    if section_ids:
        details.append(", ".join(section_ids))
    suffix = f" · {' · '.join(details)}" if details else ""
    return f"[{source.get('id', '?')}] {location}{suffix}"


def _render_response(response: dict) -> None:
    st.markdown(response.get("answer") or "No answer generated.")
    for warning in response.get("warnings", []):
        st.warning(warning)

    sources = response.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                st.markdown(f"**{_source_title(source)}**")
                st.write(source.get("text", ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], dict):
            _render_response(message["content"])
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask about your indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching sources and preparing an answer..."):
                response = ask(
                    question,
                    mode=retrieval_mode,
                    use_reranker=use_reranker,
                )
            _render_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            st.error(f"The request failed: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": {"answer": f"The request failed: {exc}"}}
            )