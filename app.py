import streamlit as st
import helper
import pickle
# import tensorflow as tf
# from keras.models import load_model  # Works with tf.keras in latest versions


try: 
    # model = load_model("duplicate_ann_model.h5")
    # model1=pickle.load(open('rf_model.pkl','rb'))
    # model2=pickle.load(open('xgb_model.pkl','rb'))
    model3=pickle.load(open('rf_w2v_model1.pkl','rb'))
    model4=pickle.load(open('xgb_w2v_model1.pkl','rb'))
    # model5=pickle.load(open('rf_bert_model.pkl','rb'))
    # model6=pickle.load(open('xgb_bert_model.pkl','rb'))
except Exception as e:
    st.error(f"Model loading failed: {e}")

st.header('Duplicate Question Detection')

q1=st.text_input('Enter question 1')
q2=st.text_input('Enter question 2')

if st.button('Predict'):
    query = helper.query_point_creator(q1, q2)
    result = model3.predict(query.reshape(1, -1))[0]

    if result > 0.5:
        st.header('These questions are duplicates')
    else:
        st.header('These questions are not duplicates')
