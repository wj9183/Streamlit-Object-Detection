import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image




def semantic_segmentation_image():


    SET_WIDTH = int(600)
    normalize_image = 1 / 255.0
    resize_image_shape = (1024, 512)


    image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)

    for image_file in image_files:

        sample_img = np.array(Image.open(image_file))
        sample_img = imutils.resize(sample_img, width=SET_WIDTH)
        
        blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0, swapRB = True, crop=False)

        cv_enet_model = cv2.dnn.readNet('models/enet/saved_model/enet-model.net')
        print(cv_enet_model)

        cv_enet_model.setInput(blob_img)

        cv_enet_model_output = cv_enet_model.forward()

        print(cv_enet_model_output.shape)

        label_values = open('models/enet/saved_model/enet-classes.txt').read().split('\n')
        label_values = label_values[ : -2+1]
        print(label_values)

        IMG_OUTPUT_SHAPE_START = 1 
        IMG_OUTPUT_SHAPE_END = 4
        classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

        class_map = np.argmax(cv_enet_model_output[0], axis = 0)

        CV_ENET_SHAPE_IMG_COLORS = open('models/enet/saved_model/enet-colors.txt').read().split('\n')

        CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[ : -2+1]

        CV_ENET_SHAPE_IMG_COLORS = np.array([np.array(color.split(',')).astype('int')  for color in CV_ENET_SHAPE_IMG_COLORS  ])

        mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

        mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1], sample_img.shape[0]) , 
                interpolation = cv2.INTER_NEAREST )

        class_map = cv2.resize(class_map, (sample_img.shape[1], sample_img.shape[0]) , 
                            interpolation=cv2.INTER_NEAREST)

        cv_enet_model_output = ( ( 0.4 * sample_img ) + (0.6 * mask_class_map) ).astype('uint8')


        my_legend = np.zeros( ( len(label_values) * 25 ,  300 , 3  )   , dtype='uint8' )
        for ( i, (class_name, img_color)) in enumerate( zip(label_values , CV_ENET_SHAPE_IMG_COLORS)) :
            color_info = [  int(color) for color in img_color  ] 
            cv2.putText(my_legend, class_name, (5, (i*25) + 17) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2 )
            cv2.rectangle(my_legend, (100, (i*25)), (300, (i*25) + 25) , tuple(color_info), -1)

        st.image(sample_img)
        st.image(cv_enet_model_output)
        st.image(my_legend)



def semantic_segmentation_video():
    st.title("Videos Object detection with SSD model")
    st.markdown("###### ※AWS ec2의 프리티어 인스턴스 성능상 상호작용 가능한 형태로 구현 불가능하여 영상으로 대체되었습니다.")
    video_file_origin = open('menu/test_video/ssd_test_video_1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
    st.video(video_file_origin)
    
    blank = """ """
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    menu = ['원본 영상', 'Object Detection']
    select = st.radio("메뉴를 골라주세요.", menu)

    if select == '원본 영상':
      video_file_origin = open('test_data/videos/origin1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
      st.video(video_file_origin)

    elif select == 'Object Detection':
        st.header("Object Detection")
        video_file_convert = open('test_data/videos/convert1.mp4', 'rb').read()
        st.video(video_file_convert)