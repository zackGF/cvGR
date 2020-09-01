#Zhu
#-*-coding:utf-8-*-
#导入模块
import numpy as np
import cv2
import math

# 打开摄像机
capture = cv2.VideoCapture(0)
VideoisOpened = capture.isOpened()

while VideoisOpened:

    # 读取帧
    flag, frame = capture.read()

    # 在特定的区域
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # crop_image=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    # 高斯滤波
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    # blur= cv2.bilateralFilter(blur,3,20.0,2.0) 
    #space是当区域半径给的是0时，用来计算区域范围的，一般情况下没用，随便给个数就行。
    #去除背景
    

    # 改变空间颜色从RGB变成HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 反底白变黑，黑变白
    # c=cv2.createBackgroundSubtractorMOG2()
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
  
    # mask2 = c.apply(mask2)
    #变换凸形态
    # kernel = np.ones((5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

    # 利用凸形态过滤背景噪声
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # 高斯滤波的阈值为127
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # 显示滤波之后的黑白照片
    cv2.imshow("Thresholded", thresh)

    # 找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # 最大区域找轮廓
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # 创建边界轮廓
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # 找出凸曲线
        hull = cv2.convexHull(contour)

        # 画出轮廓
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # 使用反正选函数找最远的点到最近的点的个数
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # 计算角度小于100度的个数
            if angle <= 100:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

        # 在矩形框上显示突起的个数
        if count_defects == 0 :
            cv2.putText(frame, "rock", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,(151,255,255),3)
        elif count_defects == 2 :
            cv2.putText(frame, "scrossor", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2,(151,255,255), 3)
        elif count_defects == 4:
            cv2.putText(frame, "paper", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,(151,255,255),3)
       
        else:
            pass
    except:
        pass

    # 展示实时帧率的图像
    cv2.imshow("Gesture", frame)
    # all_image = np.hstack((drawing, crop_image))
    # cv2.imshow('Contours', all_image)

    # 摁‘q’关闭程序
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
