# import mediapipe as mp

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Create a hand landmarker instance with the image mode:
# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path='/path/to/model.task'),
#     running_mode=VisionRunningMode.IMAGE)
# with HandLandmarker.create_from_options(options) as landmarker:
#   # The landmarker is initialized. Use it here.
#   # ...
import mediapipe as mp
import cv2
import csv
import numpy as np
import time
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
    """이미지에 핸드 랜드마크를 그리는 함수"""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # 랜드마크 연결 관계 (MediaPipe Hands 기준)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
        (5, 9), (9, 13), (13, 17)             # Palm base connections
    ]

    # Loop through the detected hands to visualize.
    for hand_landmarks in hand_landmarks_list:
        # Draw the connections
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            h, w, _ = annotated_image.shape
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            
            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Draw the landmarks
        for landmark in hand_landmarks:
            h, w, _ = annotated_image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)

    return annotated_image

def main():
    # 설정 변수
    MODEL_PATH = 'hand_landmarker.task' # 모델 파일이 필요합니다.
    INPUT_DIR = "/home/kimsaehoon/workspace/JamJamBeat/test_video" # 영상들이 있는 폴더 경로
    OUTPUT_ROOT = "dataset" # 결과가 저장될 루트 폴더
    TARGET_FPS = 30

    # Hand Landmarker 옵션 설정 (VIDEO 모드)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, # 최대 감지할 손의 개수
        min_hand_detection_confidence=0.5
    )

    # 입력 디렉토리 확인
    if not os.path.exists(INPUT_DIR):
        print(f"Error: 입력 디렉토리를 찾을 수 없습니다: {INPUT_DIR}")
        return

    # 폴더 내의 동영상 파일 리스트 가져오기
    video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"총 {len(video_files)}개의 동영상 파일을 찾았습니다.")

    # HandLandmarker 인스턴스 생성 (루프 밖에서 한 번만 생성하여 재사용)
    with HandLandmarker.create_from_options(options) as landmarker:
        
        for video_file in video_files:
            video_path = os.path.join(INPUT_DIR, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            print(f"Processing: {video_file}...")

            # 출력 디렉토리 구조 생성
            # dataset/{video_name}_infer/
            base_output_dir = os.path.join(OUTPUT_ROOT, f"{video_name}_infer")
            frames_dir = os.path.join(base_output_dir, "frames")         # 원본 프레임 저장
            landmarks_dir = os.path.join(base_output_dir, "landmarks")   # CSV 저장
            visualized_dir = os.path.join(base_output_dir, "visualized") # 시각화 이미지 저장

            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(landmarks_dir, exist_ok=True)
            os.makedirs(visualized_dir, exist_ok=True)

            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Skipping: 파일을 열 수 없습니다: {video_path}")
                continue

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 프레임 건너뛰기 간격 계산
            frame_interval = max(1, int(round(original_fps / TARGET_FPS)))
            
            frame_count = 0
            processed_frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 30FPS에 맞춰 프레임 선택
                if frame_count % frame_interval != 0:
                    frame_count += 1
                    continue

                # 1. 원본 프레임 저장 (라벨링 확인용)
                frame_filename = f"{processed_frame_count:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)

                # MediaPipe 처리를 위해 BGR -> RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # 현재 프레임의 타임스탬프 (밀리초 단위) 계산
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                
                # 추론 실행
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # 2. 랜드마크 결과 CSV 저장 (프레임마다 개별 파일)
                csv_filename = f"{processed_frame_count:06d}.csv"
                csv_path = os.path.join(landmarks_dir, csv_filename)
                
                with open(csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    # 헤더 작성
                    csv_writer.writerow(['hand_index', 'landmark_index', 'x', 'y', 'z'])
                    
                    if detection_result.hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                            for lm_idx, landmark in enumerate(hand_landmarks):
                                csv_writer.writerow([hand_idx, lm_idx, landmark.x, landmark.y, landmark.z])

                # 3. 결과 시각화 및 이미지 저장
                annotated_frame = draw_landmarks_on_image(frame, detection_result)
                cv2.imwrite(os.path.join(visualized_dir, frame_filename), annotated_frame)
                
                frame_count += 1
                processed_frame_count += 1

            cap.release()
            print(f"  -> 완료: {processed_frame_count} 프레임 처리됨.")

if __name__ == "__main__":
    main()