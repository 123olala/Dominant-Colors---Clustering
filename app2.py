import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import matplotlib.image as img 
from PIL import Image
import requests 
import cv2
from sklearn.cluster import KMeans

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
**Python libraries:** streamlit, pandas, matplotlib, seaborn, numpy, PIL, requests, sklearn.
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

class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        img_src = Image.open(requests.get(url, stream=True).raw)
        img_src = np.array(img_src)
        # percent by which the image is resized
        scale_percent = 10

        # calculate the 50 percent of original dimensions
        width = int(img_src.shape[1] * scale_percent / 100)
        height = int(img_src.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        small_img = cv2.resize(img_src, dsize)

        # convert to rgb from bgr
        img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

if len(url) > 0:
    im = Image.open(requests.get(url, stream=True).raw)
    st.write('Your image:')
    st.image(im)
    #Get colors
    dc = DominantColors(url, k) 
    colors = dc.dominantColors()
    #Display dominant colors
    st.write('Your dominant colors:')
    f, ax = plt.subplots(figsize=(7, 5))
    ax = plt.imshow([colors])
    plt.title('Dominant Colors')
    plt.axis('off')
    st.pyplot(f)





