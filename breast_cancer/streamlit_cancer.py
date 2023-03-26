# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:14:47 2023

@author: harin
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

# Loading the trained model
loaded_model = pickle.load(open("C:/Users/harin/Pictures/breast_cancer.pkl","rb"))

def cancer_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = loaded_model.predict(id_reshaped)
    
    return prediction[0]
   
    
def main():
    image = Image.open("C:/Users/harin/Pictures/cancer.jpg")
    st.image(image, width=700)
    
    st.title("BREAST CANCER PREDICTION ")

    mean_radius=st.number_input("enter the mean radius value:")
    
    mean_texture=st.number_input("enter the mean texture value:")
    
    mean_perimeter=st.number_input("enter the mean perimeter value:")
    
    mean_area=st.number_input("enter the mean area value:")
    
    mean_smoothness=st.number_input("enter the mean smoothness value:")
    
    mean_compactness=st.number_input("enter the mean compactness value:")
    
    mean_concavity=st.number_input("enter the mean concavity value:")
    
    mean_concave_point=st.number_input("enter the mean concave point value:")
    
    mean_symmetry=st.number_input("enter the mean symmentry value:")
    
    mean_fractal_dimension=st.number_input("enter the mean fractal dimension value:")
    
    radius_error=st.number_input("enter the  radius erroe value:")
    
    texture_error=st.number_input("enter the texture error value:")
    
    perimeter_error=st.number_input("enter the perimeter error value:")
    
    area_error=st.number_input("enter the area error value:")
    
    smoothness_error=st.number_input("enter the smoothness error value:")
    
    compactness_error=st.number_input("enter the compactness error value:")
    
    concavity_error=st.number_input("enter the concavity error value:")
    
    concave_point_error=st.number_input("enter the concave point error value:")
    
    symmentry_error=st.number_input("enter the symmentry error value:")
    
    fractal_dimension_error=st.number_input("enter the fractal dimension error value:")
    
    worst_radius=st.number_input("enter the worst radius value:")
    
    worst_texture=st.number_input("enter the worst texture value:")
    
    worst_perimeter=st.number_input("enter the worst perimeter value:")
    
    worst_area=st.number_input("enter the worst area value:")
    
    worst_smoothness=st.number_input("enter the worst smoothness value:")
    
    worst_compactness=st.number_input("enter the worst compactness value:")
    
    worst_concavity=st.number_input("enter the concavity value:")
    
    worst_concave_point=st.number_input("enter the concave point value:")
    
    worst_symmentry=st.number_input("enter the worst symmentry value:")
    
    worst_fractal_dimension =st.number_input("enter the worst fractal dimension value:")
    
    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = cancer_prediction([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_point,mean_symmetry,
                                     
          mean_fractal_dimension,radius_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_point_error,symmentry_error,fractal_dimension_error,
 
           worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_point,worst_symmentry,worst_fractal_dimension])
        
        st.write("The predicted cancer is:", diagnosis)
        
if __name__=='__main__':
    main()
     