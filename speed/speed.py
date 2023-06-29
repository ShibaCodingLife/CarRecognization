import cv2
import numpy as np


# 加载物体识别模型和权重
# 这里使用的是示例，具体的模型加载过程需要根据所选模型进行适配

yolo_car_path="C:/Users/Cauxer/Desktop/temp2/CarRecognization/weights/yoloCar/model_- 9 may 2023 11_18.pt"

# 定义光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化光流跟踪
prev_frame = None
trackers = []

# 创建视频捕捉对象
cap = cv2.VideoCapture('C:/Users/Cauxer/Documents/Tencent Files/2012945473/FileRecv/高清实拍高速路车流_爱给网_aigei_com.mp4')

while True:
    # 读取当前帧
    ret, frame = cap.read()
    
    if not ret:
        print('读取出错')
        break
    #cv2.imshow('Video', frame)
     # 物体识别和边界框提取
    bboxes = []  # 示例的边界框列表
    processed_frame = getmodel.run_ov_model(getmodel.get_yolo(yolo_car_path), frame)
    for  processed_frame in  processed_frame:
        bbox =  processed_frame.boxes  # 获取边界框坐标信息
    bboxes.append(bbox)
    
    
   
    
    # 如果是第一帧，则初始化光流跟踪器
    if prev_frame is None:
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = frame
        
        for bbox in bboxes:
            x, y, w, h = bbox
            bbox_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            trackers.append((bbox_pts, prev_pts))
        
        continue
    
    # 调整当前帧为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    new_trackers = []
    
    for tracker in trackers:
        bbox_pts, prev_pts = tracker
        
        # 计算光流
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        
        # 选择良好的跟踪点
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]
        
        # 绘制光流轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
    
    # 更新前一帧和前一帧的特征点
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)
    
    # 显示结果
    cv2.imshow("光流测速", frame)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
