import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import matplotlib.image as img 
from PIL import Image
import requests
from scipy.cluster.vq import whiten,vq,kmeans  

#Seed 
np.random.seed(44)

#Name of web app
st.title('FUNNY PROJECT - Finding Dominant Colors Of Your Picture')        

#Load image
nhoongj = Image.open('u.jpg')
oki44 = Image.open('a4.png')
st.image(nhoongj,width=44)

#Bios
st.sidebar.header('About me:') #Name of the sidebar header
st.sidebar.image(oki44)
st.sidebar.write('* **Name:** Long Cao (44oki)')
st.sidebar.write('* **DOB:** 20/11/2001')
st.sidebar.write('* **Github:** 123olala (Cao Long)(github.com)')

#Short_intro
st.code(''' "made by cao long ạ" ''',language='python')

st.subheader('1. Introduction:')
st.markdown("""
This project visualizes the dominant colors of an image cluster analysis (using **KMeans** unsupervised algorithm)!\n
**Python libraries:** streamlit, pandas, matplotlib, seaborn, numpy, PIL, requests, scipy.
""")    #Discription

#Instruction
st.subheader('2. Instruction:')
st.write('Copy URL of your image and paste it into the searchbox and press Enter. After that choose the number of dominant colors that you want to get. Then get the dominant colors of your image.')

#Get url and visualize image
st.subheader('3. Getting Started')
st.write('Paste your image URL here:')
url = st.text_input(label='url')
st.write('Number of dominant colors:')
k = st.select_slider(label='number colors',options=range(1,13))

st.subheader('4. Getting Result')

if len(url) > 0:
    im = Image.open(requests.get(url, stream=True).raw)
    st.write('Your image:')
    st.image(im)
    im_array = np.array(im)
    # percent by which the image is resized
    scale_percent = 10
    # calculate the 50 percent of original dimensions
    width = int(im_array.shape[1] * scale_percent / 100)
    height = int(im_array.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    small_im = im.resize(dsize, Image.ANTIALIAS)
    small_im_array = np.array(small_im)
    #Create red,green,blue array 
    r = []
    g = []
    b = []

    for row in small_im_array:
        for pixel in row:
            #A pixel contains RGB values
             if len(pixel) == 3:
                temp_r, temp_g, temp_b = pixel 
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
            else:
                temp_r, temp_g, temp_b = pixel[:3]
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
    #Dataframe with RBG values
    pixels = pd.DataFrame({'red':r,'blue':b,'green':g})
    pixels['scaled_red'] = whiten(pixels['red'])
    pixels['scaled_blue'] = whiten(pixels['blue'])
    pixels['scaled_green'] = whiten(pixels['green'])
    #Find dominant colors 
    cluster_centers,_ = kmeans(pixels[['scaled_red','scaled_blue','scaled_green']],k)
    #Get colors
    colors = []
    #Find Standard Deviations
    r_std,g_std,b_std = pixels[['red','blue','green']].std()
    #Scale actual RGB values in range of 0-1
    for cluster_center in cluster_centers:
        scaled_r,scaled_g,scaled_b = cluster_center
        colors.append((
            scaled_r * r_std/255,
            scaled_g * g_std/255,
            scaled_b * b_std/255
        ))
    #Display dominant colors
    st.write('Your dominant colors:')
    f, ax = plt.subplots(figsize=(7, 5))
    ax = plt.imshow([colors])
    plt.title('Dominant Colors')
    plt.axis('off')
    st.pyplot(f)





