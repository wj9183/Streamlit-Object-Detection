import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from models.yolo.saved_model.yolo_model import YOLO
import webbrowser


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




def about_yolo():
    text1 = """
        YOLO의 알고리즘에 대해 알아보겠습니다.  
        하나의 이미지 안에는 물체가 하나만 있을 수도 있지만, 그렇지 않을 수도 있습니다. 
        여러 개의 오브젝트를 감지하려면, 먼저 이미지를 grid로 나누어야합니다.     """

    text2 = """
        영역(grid cell)을 구분하면 영역 별로 결과가 나옵니다. 벡터가 여러 개 나온다는 겁니다.      
        최대로 그 영역의 갯수만큼 물체를 감지할 수 있게 됩니다.      
        이제 한 이미지에 대한 y의 값이 (이미지의 가로 X 세로) X 벡터에 들어있는 정보 갯수 만큼 나옵니다.      
        CNN을 마친 결과도 같은 모양으로 나옵니다.   """




    text3 = """ 
        이때 하나의 오브젝트가 여러 cell에 걸쳐있어서 bounding box가 여러 개 그려지게 될 수 있습니다.    
        벡터의 맨 첫 번째 값(pc)으로 가장 큰 값이 있는 Bounding box만 남겨놓습니다.  
        여기서 또 다른 문제가 생깁니다.  
        그럼 한 이미지에서 같이 감지된 다른 물체의 bounding box가 지워질 수 있다는 점입니다."""
        
    text4 = """
        물체 별로 가장 큰 값을 남겨두는 것이 좋을 것 같습니다.
        이때 물체 별로 구분하기 위해 만들어진 지표를 IOU(=Intersection over union)라고 합니다.  
        바운딩 박스 간의 교집합 영역의 넓이 ÷ 합집합의 영역의 넓이입니다.   
        간단히 말해서 그냥 겹치는 부분이 많은지 적은지 정도를 나타내는 것입니다.    
        IOU 값이 클수록 같은 물체일 가능성이 큽니다.    
        IOU가 높은 사각형들 중 pc값이 큰 것만 남기고 작은 것들은 싹 지우는 겁니다.  
        이걸 Non max suppression이라고 합니다.  """

    text5 = """
            grid에 여러 개의 오브젝트 중심이 있다면, 벡터 여러개를 하나의 벡터로 처리해주면 됩니다.  
            이걸 Anchor boxes 라고 합니다.   
            하나의 셀에 물체가 2개 감지될 수 있다고 하면, 하나의 cell에 2개의 벡터를 만듭니다.    
            이것이 YOLO의 알고리즘입니다."""
    
    blank = """ """
    st.title("YOLO")
    st.image("script/yolo_algorithm.png")
    st.write(text1)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.image('script/yolo_grid.png')  
    st.write(text2)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.image('script/YOLO1.JPG')
    st.write(text3)
    st.write(blank)
    st.write(blank)
    st.write(blank)    


    st.image('script/IOU1.png')
    st.image('script/IOU2.png')
    st.write(text4)
    st.write(blank)
    st.write(blank)
    st.write(blank)  
    st.image('script/anchor.png')
    st.write(text5)
    st.write(blank)
    st.write(blank)
    st.write(blank)  
    st.write(blank)
    st.write(blank)
    st.write(blank)  

    url = 'https://pjreddie.com/darknet/yolo/'

    if st.button('YOLO 공식 홈페이지'):
        webbrowser.open_new_tab(url)
    st.markdown('##### 버튼을 누르면 YOLO 공식 홈페이지로 이동합니다.')


# def yolo_image():
#     yolo = YOLO(0.6, 0.5)

#     all_classes = get_classes('models/yolo/data/coco_classes.txt')
#     image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)
#     print(image_files)
#     if image_files is not None:
#         for image_file in image_files:
#             image_np = np.array(Image.open(image_file))
#             print(image_np)


#             result_image = detect_image(image_np, yolo, all_classes)
#             st.image(result_image)

def yolo_video():
    st.title("YOLO 실행 영상")
    video_file_origin = open('menu/test_video/yolo_test_video_1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
    st.video(video_file_origin)
    st.markdown("""###### ※본 영상은 직접 촬영한 영상입니다. AWS ec2의 프리티어 인스턴스 성능상 상호작용 가능한 형태로 구현 불가능하여 영상으로 대체되었습니다. 왼쪽 사이드바에서 메뉴를 선택해주세요.""")
    blank = """ """
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    menu = ['원본 영상', 'Object Detection']
    select = st.sidebar.radio("메뉴를 골라주세요.", menu)

    if select == '원본 영상':
        st.header("원본 영상")
        video_file_origin = open('test_data/videos/origin3.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
        st.video(video_file_origin)

    elif select == 'Object Detection':
        st.header("Object Detection")
        video_file_convert = open('test_data/videos/convert3.mp4', 'rb').read()
        st.video(video_file_convert)    