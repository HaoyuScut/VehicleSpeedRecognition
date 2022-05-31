# -*- coding: utf-8 -*-
#方法1 虚拟线圈法 王浩宇 202121018517
import cv2
import numpy as np
import math
import time


# np.set_printoptions(threshold=np.inf, linewidth=850)

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

def draw(img,pt,color):
    cv2.line(img,pt[0],pt[1],color,1)
    cv2.line(img,pt[1], pt[2],color,1)
    cv2.line(img, pt[2], pt[3], color, 1)
    cv2.line(img, pt[3], pt[0], color, 1)
    return img

def distance(point1,point2):
    x1,y1 = point1
    x2,y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def Virtualcoil(img, point):  # 四点进行透视变换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    # print(h,w)
    # height = point[1][1] - point[0][1]
    # print(point)
    width = point[2][0] - point[1][0]
    height = int(distance(point[1],point[0]))

    src_list = point

    pts1 = np.float32(src_list)
    pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))

    pts3 = np.float32([[point[1][0], point[0][1]],
                       [point[1][0], point[1][1]],
                       [point[2][0],point[2][1]], [point[2][0],point[3][1]]])
    matrix_imgwarp = cv2.getPerspectiveTransform(pts1, pts3)
    img_warp = cv2.warpPerspective(img, matrix_imgwarp, (w, h))
    # cv2.imshow("Perspective transformation", img_warp)

    # cv2.waitKey(0)
    sum = 0
    for i in range(height):
         for j in range(width):
             sum += result[i][j]
    mean = sum / width / height
    return mean,result


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
            frame = LS(frame, -5, 20)
            # frame = cv2.blur(frame, (3,3))
            image = frame.copy()
            #frame = cv2.medianBlur(frame,3)

            ###########帧差求移动背景帧,标记车辆目标##############

            if background is None:
                background = frame

            nextframe = frame
            diff = cv2.absdiff(background, nextframe)
            # cv2.imshow('video', diff)
            background = nextframe  # 帧差法 背景变化
            # 结果转为灰度图
            thresh = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # 图像二值化
            thresh = cv2.threshold(thresh, 30, 255, cv2.THRESH_BINARY)[1]
            # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
            # cv2.imshow('thresh', thresh)
            kernel = np.ones((3, 3), np.float32)
            kernel1 = np.ones((9,3), np.float32)

            thresh = cv2.dilate(thresh, kernel, iterations=3)
            thresh = cv2.dilate(thresh, kernel1, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            # cv2.imshow('dilate and erode', thresh)
            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓
            Area = [0]
            Rect = []
            for c in cnts:
                # 忽略小轮廓，排除误差
                if cv2.contourArea(c) < 9000:
                    continue
                # 计算轮廓的边界框，在当前帧中画出该框
                (x, y, w, h) = cv2.boundingRect(c)
                if y < 300:
                    continue
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                Rect.append((x,y,w,h))
                contour_area = int(w * h)
                Area.append(contour_area)

            # 显示当前帧
            # cv2.imshow("frame", frame)
            max_area = max(Area)
            # print(max_area)
            # print(Rect)
            ######################################
            #四点
            # point3 = [(610, 454), (617, 509), (651, 507), (637, 454)] #目标线 第三根
            # point2 = [(487, 455), (463, 505), (473, 505), (498, 455)]  # 虚拟线圈2 中线
            # point1 = [(166, 452), (51, 503), (61, 503), (176, 452)]  # 虚拟线圈1
            # point2 = [(480, 455), (455, 505), (463, 505), (487, 455)]  # 虚拟线圈2 边线
            # point3 = [(803, 455), (826, 510), (832, 510), (809, 455)]

            point1 = [(166,451),(51,503),(77,503),(189,451)] #虚拟线圈1
            point2 = [(479, 456), (453, 506), (485, 506), (506, 456)] #虚拟线圈2
            point3 = [(800, 455), (823, 510), (829, 510), (806, 455)]#终点线圈

            # cv2.waitKey(0)

            # show a frame
            # frame = draw(frame, point1, (0, 0, 255))
            # frame = draw(frame, point2, (0, 0, 255))
            # frame = draw(frame, point3, (255, 0, 0))

            # show image
            image = draw(image, point1, (0, 0, 255))
            image = draw(image, point2, (0, 0, 255))
            image = draw(image, point3, (255, 0, 0))

            if counter == 1:
                Enter_Flag = False
                Out_Flag = False

                # Flag = True
                # print(hasEnter)
                init_gray1,init_img1 = Virtualcoil(frame, point1)
                init_gray2,init_img2 = Virtualcoil(frame, point2)
                init_gray3,init_img3 = Virtualcoil(frame, point3)
                cv2.imwrite('background.png',image)
            else:
                later_gray1,later_img1 = Virtualcoil(frame, point1)
                # cv2.imshow('later_img1',later_img1)
                later_gray2,later_img2 = Virtualcoil(frame, point2)
                #cv2.imshow('later_img2', later_img2)
                later_gray3,later_img3 = Virtualcoil(frame, point3)
                #cv2.imshow('later_img3', later_img3)

                gray_change1 = abs(later_gray1 - init_gray1)
                gray_change2 = abs(later_gray2 - init_gray2)
                gray_change3 = abs(later_gray3 - init_gray3)

                #print('第 %d'%counter,'帧',',三个线圈灰度值差为：',(gray_change1,gray_change2,gray_change3))
                if max_area > 9000:
                    # 车辆通过第一条虚拟线圈
                    if gray_change1 > 40 and Enter_Flag == False:
                        Enter_Flag = True
                        frame_enter = counter
                        print("车辆通过第一条虚拟线圈")
                        cv2.imwrite('first_pass.png', image)
                        m = 0
                    # 车辆通过第二条虚拟线圈
                    if gray_change2 > 40 and Enter_Flag == True and Out_Flag == False:
                        frame_out = counter
                        m = m + 1

                        if gray_change1 < 40:
                            Enter_Flag = False
                            Out_Flag = True
                        if m == 1:
                            n = 0
                            mid_time = (frame_out - frame_enter) / int(fps)

                            velocity = 490 / 100 / mid_time
                            v_km = 3.6 * velocity
                            time = 410 / 100 / velocity
                            print("车辆通过第二条虚拟线圈")
                            print('相差时间：%.3f' % mid_time, 's')
                            print("时速：%.3f" % velocity, "m/s，即%.3f"%v_km, 'km/h')
                            print("预计%.3f" % time, "s后撞线")
                            cv2.imwrite('second_pass.png', image)
                    # 车辆通过终点线圈
                    if gray_change3 > 40 and Out_Flag == True:
                        frame_finish = counter
                        n = n + 1
                        if n == 1:
                            end_time = (frame_finish - frame_out) / int(fps)
                            print("车辆通过终点线圈")
                            print('实际撞线时间：%.3f' % end_time, 's\n')
                            cv2.imwrite('final_pass.png', image)
                        if gray_change2 < 40:
                             Out_Flag = False



            if Enter_Flag:
                image = cv2.putText(image, "The vehicle has passed the first virtual coil", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif v_km is not None:
                image = cv2.putText(image, "The vehicle has passed the second virtual coil", (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                image = cv2.putText(image, "The time interval is "+ str('%.3f' % mid_time)+ "s", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                image = cv2.putText(image, "The speed is " + str('%.3f' % v_km) + "Km/s", (10, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                image = cv2.putText(image, "Expect the car to hit the line in " + str('%.3f' % time) + "s", (10, 95),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            if end_time is not None:
                 image = cv2.putText(image, "Actual collision time is " + str('%.3f' % end_time) + "s", (540, 35),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)




            cv2.imshow('draw', image)

            key = cv2.waitKey(1) & 0xff
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

