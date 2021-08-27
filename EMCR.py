import pickle
import numpy as np
import streamlit as st
import lightgbm

st.title("Elo Merchant Category Recommendation - Kaggle Competition with Loyalty Score Predictions")

card_dict_file = open("/content/drive/MyDrive/CASE_STUDIES/CASE_STUDY_1/DATA/card_dict.pkl", "rb")
card_dict = pickle.load(card_dict_file)
Pkl_Filename = "/content/drive/MyDrive/CASE_STUDIES/CASE_STUDY_1/DATA/LGB_Model_61579.pkl"  

def final_fun_1(card_id='C_ID_0ab67a22ab'):
  card_data=card_dict.get(card_id)
  card_data=np.array(card_data).reshape((1,-1))
  pred_y_test_pickle=lightbgm_reg_pickle.predict(card_data)
  return pred_y_test_pickle[0]

# Load the Model back from file 
with open(Pkl_Filename, 'rb') as file:
  lightbgm_reg_pickle = pickle.load(file)
  card_select = st.selectbox('Select Card ID: ',card_dict.keys())
  card_selection_msg=st.text('You have selected card: '+card_select)
  if st.button('Predict Loyalty Score'):
    loyalty_score=final_fun_1(card_select)
    st.write('Card Id: '+card_select)
    st.write("Loylty Score: "+str(loyalty_score))