import streamlit as st
# streamlit run app.py
st.title("Streamlit Demo MLOps Bootcamp Jagannath")

st.header("Heading of streamlit")

st.subheader("Sub-Heading of streamlit")

st.text("This is an example text")

st.success("success")

st.warning("warning")

st.info("can add information")

st.error("Error")

if st.checkbox("select/unselect"):
    st.text("user selected the check box")
else:
    st.text("user not selected checkbox")

state = st.radio("what is your favourite color ?",("Red","Green","Blue"))

if state== 'Blue':
    st.success("That's my favourite color")

occupation = st.selectbox("What do you do ?",['student','vlogger','employee'])

st.text(f"selected option is {occupation}")

if st.button("Example Button"):
    st.error("You clicked")

st.sidebar.header("Heading of sidebar")
st.sidebar.text("created by jagannath")