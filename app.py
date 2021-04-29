import streamlit as st
from menu.introduce import introduce
from menu.ssd import ssd_video, about_ssd
from menu.yolo import yolo_video, about_yolo
from menu.semantic_segmentation import semantic_segmentation_image, semantic_segmentation_video, about_semantic_segmentation

# import base64

# main_bg = "script/001.png"
# main_bg_ext = "jpg"

# side_bg = "script/001.png"
# side_bg_ext = "jpg"

# st.markdown(
#     f"""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#    .sidebar .sidebar-content {{
#         background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

def main():


    
    menu = ['Object Detection', 'SSD', 'YOLO', 'Semantic Segmentation']
    choice = st.sidebar.selectbox("메뉴", menu)
    
    
    if choice == 'Object Detection':
        introduce()
    elif choice == 'SSD':
        option_list = ['About SSD', 'Video']
        option = st.sidebar.selectbox('옵션을 선택하세요.', option_list)
        # if option == 'Image':
        #     ssd_image()
        if option == 'Video':
            ssd_video()
        elif option == 'About SSD':
            about_ssd()

    elif choice == 'YOLO':
        option_list = ['About YOLO', 'Video']
        option = st.sidebar.selectbox('옵션을 선택하세요.', option_list)
        # if option == 'Image':
        #     yolo_image()
        if option == 'Video':
            yolo_video()
        elif option == 'About YOLO':
            about_yolo()
    
    elif choice == 'Semantic Segmentation':
        option_list = ['About Semantic Segmentation', 'Video', 'Image']
        option = st.sidebar.selectbox('옵션을 선택하세요.', option_list)
        if option == 'Image':
            semantic_segmentation_image()
        if option == 'Video':
            semantic_segmentation_video()
        elif option == 'About Semantic Segmentation':
            about_semantic_segmentation()
    
    else:
        #메뉴를 잘못 선택한 경우 에러코드를 반환합니다.
        {"err_code" : 0}
        pass


if __name__ == '__main__':
    main()