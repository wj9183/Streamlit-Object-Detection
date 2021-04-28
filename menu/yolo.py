import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from models.yolo.saved_model.yolo_model import YOLO


def process_image(img):


  image_org = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
  image_org = np.array(image_org, dtype = 'float32')
  image_org = image_org / 255.0
  image_org = np.expand_dims(image_org, axis = 0)

  return image_org


def get_classes(file):

    with open(file) as f : 
        name_of_class = f.readlines()

    name_of_class = [class_name.strip() for class_name in name_of_class]

    return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):


    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                (top, left - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 255), 2,
                cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()

def detect_image( image, yolo, all_classes):


    pimage = process_image(image)
    #3개를 예측해준다.
    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)
    if image_boxes is not None: #비어있지 않으면
        box_draw(image, image_boxes, image_scores, image_classes, all_classes )
    return image


def yolo_image():
    yolo = YOLO(0.6, 0.5)

    all_classes = get_classes('models/yolo/data/coco_classes.txt')
    image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)
    print(image_files)
    if image_files is not None:
        for image_file in image_files:
            image_np = np.array(Image.open(image_file))
            print(image_np)


            result_image = detect_image(image_np, yolo, all_classes)
            st.image(result_image)

def yolo_video():
    st.title("YOLO model")
    st.markdown("##### ※AWS ec2의 프리티어 인스턴스 성능상 상호작용 가능한 형태로 구현 불가능하여 영상으로 대체되었습니다.")
    video_file_origin = open('menu/test_video/yolo_test_video_1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
    st.video(video_file_origin)


    st.header("원본")
    video_file_origin = open('test_data/videos/origin3.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
    st.video(video_file_origin)

    if st.button('변환'):
        st.header("Object Detection")
        video_file_convert = open('test_data/videos/convert3.mp4', 'rb').read()
        st.video(video_file_convert)    