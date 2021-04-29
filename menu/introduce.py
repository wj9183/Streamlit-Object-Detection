import streamlit as st


def introduce():

    title = "Object Detection"
    text1 = """    
            Object Detection은 컴퓨터 비전과
            이미지 처리와 관련된 컴퓨터 기술로서, 디지털 이미지와 비디오로
            인간, 건물, 자동차 등을 감지하는 일을 다룹니다. 
            잘 연구된 분야로는 얼굴 검출, 보행자 검출이 포함되며, 영상 복구, 비디오 감시를 포함한 수많은 컴퓨터 비전 분야에 응용되고 있습니다.  
            물체가 무엇인가(Image classification)와 그 물체가 이미지의 어느 위치에 있는지(Object localization)를 동시에 파악하는 것이 Object detection입니다.   """
    blank = """ """
    text2 = """Object detection에 앞서 CNN에 대해 얘기해보겠습니다.    
        특정 물체를 감지하는 커널을 이미지를 윗부분부터 훑습니다.   
        이렇게 커널을 가지고 이미지를 Convolution하는 것을 Sliding window 방식이라고 합니다."""
    
    text3 = """         
                그리고 그 결과를 가지고 히스토그램을 그리는데, 이를 HOG(Histogram of oriented gradients)라고 합니다.
                비슷한 히스토그램이 그려지는 부분에 그 대상이 있다고 판단하는 것입니다.
                                
                                    
                                        """
        



    text4 = """원본 이미지가 있으면, 스케일을 달리하여 여러 개의 이미지로 만듭니다.     
        그렇게 이미지들이 스케일 별로 늘어서있는 것을 이미지 피라미드라고 합니다.           
        그 모든 이미지를 같은 사이즈의 커널로 훑습니다.  
        큰 이미지를 Convolution 했을 때엔 커널 사이즈가 작아서 물체를 찾아내지 못할 수 있지만,
        이미지를 작게 만들면 같은 사이즈의 커널로 찾아낼 가능성이 높아집니다.   
        여기까지가 CNN의 방식입니다."""

    text5= """그럼 한 원본 이미지가 주어졌을 때 그 안에 있는 여러 물체를 감지하려면, 서로 다른 물체를 감지하는 여러 개의 커널을 사용해야할 것입니다.  
            그리고 그 모든 커널로 모든 사이즈의 이미지들을 훑어야합니다.    
            시간이 오래 걸릴 수 밖에 없습니다.
            그러면 여기서 의문이 듭니다."""


    text6=  """그럼 한 번 CNN을 할 때 여러 물체를 한 번에 잡을 수는 없는 걸까요?"""

    text7 =  """그런 아이디어로 만들어진 게 SSD와 YOLO입니다.   
                이미지 피라미드 만들고 특정 물체를 감지하는 커널들로 전부 훑고 이런 과정을 거치지 않고,
                한 번 CNN이 수행되는 동안 모든 물체를 감지합니다.   
                그러니 속도가 빠를 수 밖에 없고, 자율주행 등을 포함해 더 폭넓은 분야에서 활용될 수 있는 것입니다.     
                이 앱의 기능을 통해 SSD와 YOLO, 그리고 Semantic Segmentation에 대해 알아보겠습니다."""
 
 
    st.title(title)
    st.image('script/objectdetection.jpg')
    st.write(text1)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.title("Convolution Neural Network")
    st.markdown("![Alt Text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fxu5oD%2Fbtq3KRyEh0j%2Fm2VtVsMQ6kZrbU9TkPyqS1%2Fimg.gif)")
    st.write(text2)
    st.image('script/HOG1.jpg')
    st.image('script/HOG2.jpg')
    st.write(text3)
    st.image('script/pyramid.png')
    st.write(text4)
    st.image('script/CNN.png')
    st.write(text5)
    st.header(text6)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)
    st.write(blank)

    st.title("SSD, YOLO, Semantic Segmentation")
    st.image('script/deep_learning_object_detection_history.png')
    st.write(text7)
