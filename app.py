import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
st.title("News Text Style Classifier")
x=["official agency published annual report","council approved public budget","shocking secret miracle guaranteed","unbelievable claim share immediately"]
y=["Routine-style","Routine-style","Sensational-style","Sensational-style"]
model=make_pipeline(TfidfVectorizer(),LogisticRegression()).fit(x,y)
t=st.text_area("Enter headline")
if st.button("Analyze") and t.strip(): st.write(model.predict([t])[0])
st.warning("Style classification does not establish whether a news story is true.")
