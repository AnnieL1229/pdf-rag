import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Demo", layout="wide")
st.title("PDF RAG Demo")
st.caption("Upload a few PDFs, then ask questions against the indexed content.")

with st.sidebar:
    st.header("API")
    api_base_url = st.text_input("Base URL", value=API_BASE_URL)


st.subheader("1. Upload PDFs")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    accept_multiple_files=True,
    type=["pdf"],
)

if st.button("Ingest files", use_container_width=True):
    if not uploaded_files:
        st.warning("Choose at least one PDF first.")
    else:
        try:
            files = [
                ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
                for uploaded_file in uploaded_files
            ]
            response = requests.post(f"{api_base_url}/ingest", files=files, timeout=120)
            if response.ok:
                st.success("Ingestion completed.")
                st.json(response.json())
            else:
                st.error(response.text)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")


st.subheader("2. Ask a question")
question = st.text_input("Question", placeholder="Ask a question about your uploaded PDFs…")

if st.button("Submit question", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            response = requests.post(
                f"{api_base_url}/query",
                json={"question": question},
                timeout=120,
            )
            if response.ok:
                payload = response.json()
                st.markdown("### Answer")
                if payload.get("needs_clarification"):
                    st.warning(payload.get("clarification_question") or "Please clarify your question.")
                elif payload.get("answer"):
                    st.write(payload["answer"])
                else:
                    st.info("No answer returned.")
                if payload.get("validation_reason") and (
                    payload.get("needs_clarification")
                    or payload.get("coverage_sufficient") is False
                    or payload.get("insufficient_evidence")
                ):
                    st.caption(f"Validation: {payload['validation_reason']}")

                if payload.get("refusal_reason"):
                    st.warning(payload["refusal_reason"])

                st.markdown("### Retrieved sources")
                sources = payload.get("retrieved_chunks") or payload.get("citations") or []
                if sources:
                    for source in sources:
                        score = source["final_score"] if source["final_score"] is not None else 0.0
                        with st.expander(
                            f"{source['filename']} - page {source['page_number']} - score {score:.3f}"
                        ):
                            st.write(source["text"])
                else:
                    if payload.get("insufficient_evidence"):
                        st.info("Retrieved evidence was too weak to answer confidently.")
                    else:
                        st.info("No sources returned.")
            else:
                st.error(response.text)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
