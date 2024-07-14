import cv2
import time  # 지연을 위해 time 모듈을 임포트합니다.

def capture_image_from_webcam(filename='1m_1.jpg'):
    cap = cv2.VideoCapture(0)  # 웹캠 인덱스 조정

    if not cap.isOpened():
        print("Error: 웹캠을 열 수 없습니다.")
        return

    time.sleep(2)  # 웹캠이 완전히 초기화될 시간을 줍니다.

    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
    else:
        cv2.imwrite(filename, frame)
        print(f"{filename}으로 이미지가 저장되었습니다.")

    cap.release()

if __name__ == "__main__":
    capture_image_from_webcam()
