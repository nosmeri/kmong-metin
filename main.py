import glob
from ultralytics import YOLO
import os

# YOLOv8n 모델 로드
model = YOLO('./train/weights/best.pt')

# 이미지 파일 경로 리스트 가져오기
image_paths = glob.glob('imgs/*')

# 결과 저장 디렉토리
results_dir = 'results'

for image_path in image_paths:
    # 이미지에서 객체 감지 수행
    results = model(image_path)

    # 결과 저장
    base_filename = os.path.basename(image_path)
    new_filename = os.path.splitext(base_filename)[0] + '_result.jpg'
    results[0].save(filename=os.path.join(results_dir, new_filename))

print("모든 이미지 처리가 완료되었습니다.")
