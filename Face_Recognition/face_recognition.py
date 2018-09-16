import cv2

# cascade를 불러온다.
# cascade는 얼굴로부터 다른 부분을 추적하기 위해 적용하는 여러 필터다.
face_cascade = cv2.CascadeClassifier("/Users/khosungpil/udemy/Deep Learning and Computer Vision/Module_1_Face_Recognition/data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/khosungpil/udemy/Deep Learning and Computer Vision/Module_1_Face_Recognition/data/haarcascade_eye.xml")

# single image에서 사각형을 그리기 위해
def detect(gray, frame):
    """
        gray : 흑백사진을 argument
        frame : 원본 사진의 프레임
    """
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    """
        1.3 : scale factor, 사진의 이미지가 얼마나 커지고 작아지는지
        또는 얼만큼 filter의 사이즈가 커널을 크게 만드는지
        5 : minNeighbors, 픽셀 영역을 허용하기 위해 하나 이상의 5개 인접 영역도 허용한다.
        1.3과 5일때가 가장 결과가 좋다.
    """
    for(x, y, w, h) in faces:
        """
            x, y : 좌측상단에서부터 사각형이 얼굴을 추적하기 시작한다.
            w, h : 사각형의 높이와 너비
        """
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        """
            cv2.rectangle
        Args:
            frame : input image
        """
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # roi : region of interest(사각형 내부의..) 두 개의 roi가 필요
        # 하나는 흑백 하나는 컬러
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # 얼굴에서 눈을 추적하기 위해 새로운 eye에 대한 for loop 지정
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # 반환하는 값은 얼굴과 눈을 추적한 사각형의 original frame을 반환한다.
    return frame

video_capture = cv2.VideoCapture(0)
"""
Args:
    0: 내부카메라 사용 시 0, 외부 카메라 사용 시 1
"""
while True:
    # _ : 일단 언패킹 시에 첫 번째 값은 무시한다.
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 일단 이미지를 흑백으로 바꿔 이미지처리가 쉽게 하도록 한다.
    
    canvas = detect(gray, frame)
    
    # 지금부터는 연속적인 output으로 움직이는 장면을 보여주기 위해 트릭을 이용
    cv2.imshow('Video', canvas)

    # q를 눌렀을 때 종료한다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# webcam을 끄기 위해 release method 호출
video_capture.release()

# 창을 없애기 위해
cv2.destroyAllWindows()