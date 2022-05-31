#方法1 虚拟线圈法 王浩宇 202121018517
import cv2
import cv2.cv2
import numpy as np
import math
import time


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
    output_image = (output_image * 255.0).astype(np.uint8)
    return output_image

if __name__ == '__main__':
    cap = cv2.VideoCapture('final.mov')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取输入视频的帧率
    counter = 0
    print(fps)
    background = None
    v_km = None
    end_time = None
    while cap.isOpened():
        counter += 1
        # get a frame
        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(frame, (960, 540))  # 调整大小
            if counter == 1:
                cv2.imwrite('origin.png',frame)
                frame = LS(frame, -5, 20)
                cv2.imwrite('LS.png', frame)
                frame = cv2.medianBlur(frame, 3)
                cv2.imwrite('Blur.png', frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()