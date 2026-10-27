import streamlit as st

from src.rag_pipeline import ChatAnswer, MedicalRAGChatbot


st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_chatbot() -> MedicalRAGChatbot:
    return MedicalRAGChatbot()


def _show_sources(response: ChatAnswer) -> None:
    if not response.sources:
        return

    with st.expander("Sources used"):
        for source in response.sources:
            st.markdown(f"- **[{source['id']}] {source['name']}**")


st.title("🏥 Medical RAG Assistant")
st.caption("Ask general health questions grounded in the documents available to this assistant.")

with st.sidebar:
    st.subheader("About this assistant")
    st.write(
        "Answers are generated from the local medical reference documents and are "
        "not a substitute for professional medical advice."
    )
    st.warning(
        "For severe or rapidly worsening symptoms, contact your local emergency "
        "services or seek urgent medical care."
    )
    st.caption("Avoid entering names, contact details, or other personal health information.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- **[{source['id']}] {source['name']}**")

query = st.chat_input("Ask a general medical question")
if query:
    query = query.strip()
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        try:
            with st.spinner("Searching the reference documents and preparing an answer..."):
                response = get_chatbot().ask_question(query, history)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(
                "The assistant could not complete your request. Confirm Ollama and "
                "the required local models are available, then try again."
            )
            with st.expander("Technical details"):
                st.exception(exc)
        else:
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    },
                ]
            )
            with st.chat_message("assistant"):
                st.markdown(response.answer)
                _show_sources(response)
