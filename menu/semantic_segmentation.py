import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


def about_semantic_segmentation():
    what_is_semantic_1 = """
                            Semantic segmentation은 Object detection과 조금 다른 개념입니다.    
                            Object Detection은 물체가 있는 위치를 찾고 Boxing을 하는 것이고, 
                            Segmentation은 Image를 Pixel로 나눠서 각 pixel이 어떤 물체 class인지 구분하는 것입니다."""
                            
    what_is_semantic_2 = """
                            Semantic segmentation은 이미지에 있는 모든 Pixel을 분류합니다.  
                            One Hot encoding을 통해 각 class에 대한 채널을 만듭니다.    
                            Class의 갯수 만큼 만들어진 채널을 argmax 함수를 이용해 하나의 결과물을 돌려줍니다."""
    what_is_semantic_3 ="""
                            Semantic segmentation은 같은 class에 속하는 object 끼리는 구분하지 않습니다.    
                            같은 class에 속하는 것들 끼리도 구분해주는 것은 instance segmentation이라고 합니다. 
                            Downsampling: 주 목적은 차원을 줄여서 적은 메모리로 깊은 Convolution 을 할 수 있게 하는 것입니다.   
                             보통 stride 를 2 이상으로 하는 Convolution 을 사용하거나, pooling을 사용합니다. 이 과정을 진행하면 어쩔 수 없이 feature 의 정보를 잃게됩니다.
마지막에 Fully-Connected Layer를 넣지 않고, Fully Connected Network 를 주로 사용합니다. FCN 모델에서 위와같은 방법을 제시한 후 이후에 나온 대부분의 모델들에서 사용하는 방법입니다.
Upsampling: Downsampling 을 통해서 받은 결과의 차원을 늘려서 인풋과 같은 차원으로 만들어 주는 과정입니다. 주로 Strided Transpose Convolution 을 사용합니다."""



    st.title('About Semantic segmentation')
    blank = """ """
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.image("script/differenceofsemantic.png")
    
    st.write(what_is_semantic_1)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.image("script/channel.png")
    st.write(what_is_semantic_2)

    st.write(what_is_semantic_3)






# def semantic_segmentation_image():


#     SET_WIDTH = int(600)
#     normalize_image = 1 / 255.0
#     resize_image_shape = (1024, 512)


#     image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)

#     for image_file in image_files:

#         sample_img = np.array(Image.open(image_file))
#         sample_img = imutils.resize(sample_img, width=SET_WIDTH)
        
#         blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0, swapRB = True, crop=False)

#         cv_enet_model = cv2.dnn.readNet('models/enet/saved_model/enet-model.net')
#         print(cv_enet_model)

#         cv_enet_model.setInput(blob_img)

#         cv_enet_model_output = cv_enet_model.forward()

#         print(cv_enet_model_output.shape)

#         label_values = open('models/enet/saved_model/enet-classes.txt').read().split('\n')
#         label_values = label_values[ : -2+1]
#         print(label_values)

#         IMG_OUTPUT_SHAPE_START = 1 
#         IMG_OUTPUT_SHAPE_END = 4
#         classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

#         class_map = np.argmax(cv_enet_model_output[0], axis = 0)

#         CV_ENET_SHAPE_IMG_COLORS = open('models/enet/saved_model/enet-colors.txt').read().split('\n')

#         CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[ : -2+1]

#         CV_ENET_SHAPE_IMG_COLORS = np.array([np.array(color.split(',')).astype('int')  for color in CV_ENET_SHAPE_IMG_COLORS  ])

#         mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

#         mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1], sample_img.shape[0]) , 
#                 interpolation = cv2.INTER_NEAREST )

#         class_map = cv2.resize(class_map, (sample_img.shape[1], sample_img.shape[0]) , 
#                             interpolation=cv2.INTER_NEAREST)

#         cv_enet_model_output = ( ( 0.4 * sample_img ) + (0.6 * mask_class_map) ).astype('uint8')


#         my_legend = np.zeros( ( len(label_values) * 25 ,  300 , 3  )   , dtype='uint8' )
#         for ( i, (class_name, img_color)) in enumerate( zip(label_values , CV_ENET_SHAPE_IMG_COLORS)) :
#             color_info = [  int(color) for color in img_color  ] 
#             cv2.putText(my_legend, class_name, (5, (i*25) + 17) , 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2 )
#             cv2.rectangle(my_legend, (100, (i*25)), (300, (i*25) + 25) , tuple(color_info), -1)

#         st.image(sample_img)
#         st.image(cv_enet_model_output)
#         st.image(my_legend)



def semantic_segmentation_video():

    menu = ['Semantic Segmentation Test 영상', '원본 영상', 'Segmentation']
    select = st.sidebar.radio("메뉴를 골라주세요.", menu)

    if select == 'Semantic Segmentation Test 영상':
        st.title("Semantic Segmentation Test 영상")
        blank = """ """
        st.write(blank)
        st.write(blank)
        video_file_origin = open('menu/test_video/semantic_test_video_1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
        st.video(video_file_origin)
        st.markdown("""###### ※본 영상은 직접 촬영한 영상입니다. AWS ec2의 프리티어 인스턴스 성능상 상호작용 가능한 형태로 구현 불가능하여 영상으로 대체되었습니다.""")
            
    if select == '원본 영상':
        st.title("원본 영상")
        video_file_origin = open('test_data/videos/origin4.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
        st.video(video_file_origin)

    elif select == 'Segmentation':
        st.title("Segmentation")
        video_file_convert = open('test_data/videos/convert4.mp4', 'rb').read()
        st.video(video_file_convert)