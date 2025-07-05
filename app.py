import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import numpy as np
import tensorflow as tf
import cv2
import queue
from streamlit_autorefresh import st_autorefresh


MODEL_PATH = "model.tflite"


@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


class TFLiteVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.interpreter = load_tflite_model(MODEL_PATH)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.result_queue = queue.Queue(maxsize=1)  # 최대 1개의 결과만 저장

    def run_inference(self, img):
        input_shape = self.input_details[0]["shape"]
        resized = cv2.resize(img, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output_data

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # flipped = img[:, ::-1, :]
        result = self.run_inference(img)
        label = int(np.argmax(result))

        # print("🐍 File: mask_ai/app.py | Line: 40 | recv ~ label1111111111111", label)
        try:
            self.result_queue.put_nowait(label)
        except queue.Full:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("웹캠 실시간 TFLite 인퍼런스")
st.write("아래 Start 버튼을 눌러 웹캠 영상을 확인하세요.")

st_autorefresh(interval=500, key="autorefresh")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=TFLiteVideoProcessor,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    result_placeholder = st.empty()
    if webrtc_ctx.state.playing:
        try:
            label = webrtc_ctx.video_processor.result_queue.get_nowait()
            # print("🐍 File: mask_ai/app.py | Line: 68 | undefined ~ label", label)
            result_placeholder.info(f"예측 결과: {label}")
        except queue.Empty:
            result_placeholder.info("예측 결과 대기 중...")
