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
    script_what_is_ssd_1 = """SSD는 Single Shot Detector의 줄임말로, 실시간 물체 감지를 목적으로 설계된 모델입니다.     
                            동시에, 정확성에서 높은 성능을 보이는 Faster R-CNN에 앞서는 정확성도 가지고 있습니다.
                            """
    st.write(script_what_is_ssd_1)
    st.image("script/inference.png")

    script_what_is_ssd_2 = """ssd는 multi scale feature와 default box를 사용하고 이미지의 해상도를 떨어 뜨려 속도를 향상시켰습니다.
                                이를 통해 SSD는 손실 없는 높은 정확도로 물체를 감지할 수 있습니다."""
    st.write(script_what_is_ssd_2)
    # st.image("script/ssd_layers.jpeg")
    script_what_is_ssd_3 = """ssd의 핵심이 되는 아이디어는, Feature Map이 Convolution 연산을 거치면서 크기가 점점 작아진다는 점을 이용한 것입니다.
                            RPN에서 Anchor라고 부르는 것과 같은 기능을 하는 Default Box라는 것을 두고,
                            큰 Feature Map에서는 작은 물체를 검출하고, 작은 Feature Map에서는 큰 물체를 검출하는 것입니다."""
    st.write(script_what_is_ssd_3)





# def ssd_image():
    
#   image_files = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=True)

#   if image_files is not None:
    
#     model = keras.models.load_model("./models/ssd/saved_model/")
#     for image_file in image_files:
#       show_inference(model, image_file)




def ssd_video():
    st.title("SSD")
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