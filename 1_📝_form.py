import streamlit as st

st.set_page_config(
    page_title="PatIt: Create your Patent",
    page_icon="📝",
)

questions = []
with st.form("patent"):
    questions.append(
        st.text_area("What is the technical field of your invention?", height=100)
    )
    questions.append(
        st.text_area(
            "What the technical problem does this invention deal with? ", height=100
        )
    )
    questions.append(
        st.text_area(
            "If you know some solutions available on the market, please give a brief description and their "
            "inconvenient",
            height=100,
        )
    )
    questions.append(
        st.text_area(
            "what the improvement made by your invention comparing to the available solutions?",
            height=100,
        )
    )
    questions.append(
        st.text_area(
            "please give a detailed description on different embodiments of your invention",
            height=100,
        )
    )

    submit = st.form_submit_button("Send")

if submit:
    st.session_state["answers"] = "\n".join(questions)
