import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model

# Load All Files
model = load_model('classification_model')

def run():
    # membuat title
    st.title('CUSTOMER COMPLAINTS PREDICTION')

    # tambah deskripsi
    st.write('MILESTONES 2 PHASE 2')
    st.write('Made by: Vincent Timothy Djaja')
    st.write('Batch: dsft-rmt-017')

    # tambah gambar
    image = Image.open('cs.jpg')
    st.image(image, caption='Bank teller')

    with st.form(key='form_parameters'):
      cp = st.text_input('Write down your complaints here please', 
      value='loan relatively new loan refinanced va loan escrow account lender required pay home\
         owner insurance property tax received property tax delinquent notice winter tax lender \
          attributed timing loan sure paid either previous lender current lender either way get \
            paid received another property tax delinquent notice next round property tax due lender\
               lakeview loan servicing llc')

      st.markdown('---')

      submitted = st.form_submit_button('Predict')

   
    data_inf = {'Write down your complaints here please':cp} 
    data_inf = pd.DataFrame([data_inf])
    
    if submitted:
            
      # Predict using Neural Network
      y_pred_inf = model.predict(data_inf).argmax(axis=1)
      
      
      st.write('# Predict : ', str(int(y_pred_inf)))
      st.markdown('---')
      st.write('Notes:')
      st.write('1 = Debt Collection')
      st.write('2 = Credit Reporting')
      st.write('3 = Credit Card')
      st.write('4 = Retail Banking')
      st.write('5 = Mortgages and Loans')
      

if __name__ == '__main__':
    run()