import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("웹캠 실시간 스트리밍")
st.write("아래 Start 버튼을 눌러 웹캠 영상을 확인하세요.")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # # 예시: 영상을 상하 반전
    # flipped = img[::-1, :, :]
    # 예시: 영상을 좌우 반전
    flipped = img[:, ::-1, :]
    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
