import numpy as np
import tensorflow as tf

MODEL_PATH = "model.tflite"  # 테스트할 tflite 파일 경로

# 1. 모델 로드 및 텐서 할당
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 2. 입력/출력 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("입력 정보:", input_details)
print("출력 정보:", output_details)

# 3. 입력 데이터 타입 및 shape에 맞게 더미 데이터 생성
input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

# float32 모델 예시
if input_dtype == np.float32:
    # 예: 0~1 사이 float32 값
    input_data = np.random.random_sample(input_shape).astype(np.float32)
# uint8 모델 예시 (양자화 모델)
elif input_dtype == np.uint8:
    # 예: 0~255 사이 uint8 값
    input_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
# int8 모델 예시
elif input_dtype == np.int8:
    input_data = np.random.randint(-128, 128, size=input_shape, dtype=np.int8)
else:
    raise ValueError(f"지원하지 않는 입력 dtype: {input_dtype}")

print("입력 데이터 shape:", input_data.shape, "dtype:", input_data.dtype)

# 4. 입력 텐서에 데이터
