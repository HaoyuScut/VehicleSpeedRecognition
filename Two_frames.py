#方法二
import cv2

import numpy as np

#亮度和饱和度调整
def LS(input_image,lightness,saturation):
    input_image = input_image.astype(np.float32) / 255.0
    hlsImg = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
    MAX_VALUE = 100
    hlsCopy = np.copy(hlsImg)
    # 1.调整亮度（线性变换)
    hlsCopy[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
    # 饱和度
    hlsCopy[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # HLS2BGR
    output_image = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)

    return output_image


if __name__ == '__main__':
    cap = cv2.VideoCapture('final.mov')
    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (960, 540))  # 调整大小
    # prevframe = frame  # 第一帧
    background = None
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))  # 调整大小
        cv2.imwrite('frame.png', frame)
        image = frame.copy()
        if background is None:
            background = frame
            continue
        nextframe = frame
        if ret:
            diff = cv2.absdiff(background, nextframe)
            # cv2.imshow('video', diff)
            background = nextframe  # 帧差法 背景变化
            # 结果转为灰度图
            thresh = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 图像二值化
            thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1]
            # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
            kernel = np.ones((3, 3), np.float32)
            kernel1 = np.ones((3, 6), np.float32)

            thresh = cv2.dilate(thresh, kernel,iterations=3)
            thresh = cv2.dilate(thresh, kernel1, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            # cv2.imshow('thresh', thresh)
            # cv2.waitKey(0)


            # 阀值图像上的轮廓位置

            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print('cnts',cnts)

            # 遍历轮廓
            Area = [0]

            for c in cnts:
                # 忽略小轮廓，排除误差
                if cv2.contourArea(c) < 8000:
                    continue
                # 计算轮廓的边界框，在当前帧中画出该框
                (x, y, w, h) = cv2.boundingRect(c)
                if y < 300:
                    continue

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                contour_area = int(w*h)
                Area.append(contour_area)

            # 显示当前帧
            cv2.imshow("frame", image)
            max_area = max(Area)
            print(max_area)



            # cv2.imshow("thresh", thresh)

            # cv2.imshow("threst2", thresh2)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

