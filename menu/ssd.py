import streamlit as st
import cv2
import tensorflow as tf
import os
import pathlib
import keras

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import webbrowser

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util




utils_ops.tf = tf.compat.v1

tf.gfile = tf.io.gfile

PATH_TO_LABELS = "./models/ssd/saved_model/mscoco_label_map.pbtxt"

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




def load_image(image_file):
  img = Image.open(image_file)
  return img


def run_inference_for_single_image(model, image):
  # 넘파이 어레이로 바꿔준다.
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}

  # print(output_dict)
  
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
    output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])  
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_file):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.

  image_np = np.array(Image.open(image_file))
  image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  # image_np = cv2.imread(str(image_path))
  # print(image_np)
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  print(output_dict)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.array(output_dict['detection_boxes']),
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed',None),
      use_normalized_coordinates=True,
      line_thickness=8)
  image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
  scaleX = 0.6
  scaleY = 0.6
  image_np_bgr_resize = cv2.resize(image_np_bgr, None, fx = scaleX, fy=scaleY, interpolation = cv2.INTER_LINEAR)
  st.image(image_np_bgr_resize)




def about_ssd():
    st.title("About SSD")
    script_what_is_ssd_1 = """
                              SSD는 YOLO보다 1년 가량 먼저 만들어졌습니다.  
                              YOLO가 SSD를 많이 차용하여, 서로 닮은 부분이 많습니다.  
                              성능도 각자 버전을 업데이트를 거듭하면서 우열관계가 뒤집히길 반복했습니다."""

    script_what_is_ssd_2 ="""
                              SSD의 핵심은 Image pyramid에 있습니다.  
                              한 번 CNN하는 과정에서 전부 Image Pyramid로 처리합니다.   
                              그에 반해 Sliding window는 느립니다.  
                              Image Pyramid를 만들어두고 각 이미지를 다 CNN 해야하기 때문입니다."""
                              
    script_what_is_ssd_3 ="""
                              CNN 과정을 먼저 생각해봅니다. 
                              여러 개의 커널(피쳐)들이 이미지를 Convolution합니다.  
                              그러면서 이미지 사이즈가 작아지고, 이미지가 커널의 갯수만큼 많아집니다. 
                              Convolution을 했으니 이제 Activation을 하고, Pooling을 합니다.  
                              그럼 또 이미지 사이즈가 줄어듭니다. 
                              위 과정을 반복하다보면, 이미지가 계속해서 작아집니다."""



    script_what_is_ssd_4 ="""
                              Image Pyramid를 굳이 만들지 않아도, CNN 과정에 이미 포함되어있다는 것입니다.  
                              이미지 사이즈가 축소가 된다는 건, 이미지가 확대된다는 것과 같은 의미입니다.  
                              말이 참 모순적인데, 커널 사이즈는 그대로이기 때문입니다.  
                              이미지가 작아지면 이미지 안에 있던 물체가 똑같은 커널의 한 프레임 안에 다 들어올 수 있게 됩니다."""


    script_what_is_ssd_5 ="""
                              CNN의 과정을 똑같이 하면서,
                              Convolution을 하고 Pooling을 할 때 그 모든 결과를 모읍니다.
                              엄청나게 많은 결과가 나옵니다.

                              YOLO와 같이, Bounding box가 엄청 많이 나오게 될 겁니다.
                              Pc가 가장 큰 값으로 NMS(Non max suppression) 해서, 각 오브젝트마다 하나씩의 Bounding box만 남깁니다.
                              여기까지가 ssd의 알고리즘입니다.

                              정리하면 ssd는, image pyramid를 CNN의 본래 convolution하는 과정에 자연스럽게 넣어서,
                              여러개의 이미지 확대 효과를 본 걸 grid cell 나눠서 object를 detection한 것입니다.
                            """
    st.image("script/inference.png")
    
    st.write(script_what_is_ssd_1)
    blank = """ """
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.image("script/pyramid.png")
    st.write(script_what_is_ssd_2)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.image("script/pooling.png")

    st.write(script_what_is_ssd_3)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.image("script/ssdarchitecture.png")
    st.write(script_what_is_ssd_4)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.image("script/NMS.png")
    st.write(script_what_is_ssd_5)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)    
    url = 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md'

    if st.button('TensorFlow 2 Detection Model Zoo'):
        webbrowser.open_new_tab(url)
    st.markdown('##### 버튼을 누르면 TensorFlow 2 Detection Model Zoo로 이동합니다.')


# def ssd_image():
    
#   image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)

#   if image_files is not None:
    
#     model = keras.models.load_model("./models/ssd/saved_model/")
#     for image_file in image_files:
#       show_inference(model, image_file)




def ssd_video():

    menu = ['SSD Test 영상','원본 영상', 'Object Detection']
    select = st.sidebar.radio("메뉴를 골라주세요.", menu)
    if select == 'SSD Test 영상':
      st.title("SSD Test 영상")
      blank = """ """
      st.write(blank)
      st.write(blank)
      video_file_origin = open('menu/test_video/ssd_test_video_1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
      st.video(video_file_origin)
      st.markdown("""###### ※본 영상은 직접 촬영한 영상입니다. AWS ec2의 프리티어 인스턴스 성능상 상호작용 가능한 형태로 구현 불가능하여 영상으로 대체되었습니다. 왼쪽 사이드바에서 메뉴를 선택해주세요.""")
            
    if select == '원본 영상':
      st.title("원본 영상")
      video_file_origin = open('test_data/videos/origin1.mp4', 'rb').read()     #비디오 파일 읽어와라. 'rb'(어떤 용도로 읽어올 건지) 안써주면 안됨.
      st.video(video_file_origin)

    elif select == 'Object Detection':
        st.title("Object Detection")
        video_file_convert = open('test_data/videos/convert1.mp4', 'rb').read()
        st.video(video_file_convert)