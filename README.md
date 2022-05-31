# VehicleSpeedRecognition
Vehicle speed recognition and crash prediction based on virtual coil method 

基于虚拟线圈法的车速识别和撞线预测

华南理工大学 模式识别实践

觉得有帮助可以点下Star

## 最终效果图
![img](https://github.com/HaoyuScut/VehicleSpeedRecognition/blob/master/Process_files/result1.png)
![img](https://github.com/HaoyuScut/VehicleSpeedRecognition/blob/master/Process_files/result2.png)
![img](https://github.com/HaoyuScut/VehicleSpeedRecognition/blob/master/Process_files/result3.png)

## 设计思路
项目的编程环境为python3.8，编译器使用pycharm x64，视频序列4k60帧每秒。项目采用帧差法识别车辆、虚拟线圈法估算车速，取线圈内平均灰度值相对于没有车辆的线圈内平均灰度值的变化作为对象特征，当差值的绝对值大于某一阈值时，判断有汽车通过线圈。

另外附有检测虚拟线圈四角点的代码 Mouse_point.py

直接运行，请运行main.py，要求有opencv库

## 基于虚拟线圈的车速检测算法
![img](https://github.com/HaoyuScut/VehicleSpeedRecognition/blob/master/Process_files/result.png)

## 总结
三辆车实际撞线时间均略小于预计撞线时间，对此进行分析，有以下几种可能性：

1.车辆通过路口之后会进行加速，第二辆车尤为明显，导致实际的时间短于预计。

2.实际的撞线距离和计算时采用的撞线距离有一定的误差，比如虚拟线圈发生灰度突变时，车辆已经相对于虚拟线圈的边界向里驶入了一些，导致实际的距离小于算法中给出的距离，影响到撞线时间。

3.计算速度时，两虚拟线圈之间的间隔距离和实际比有一定误差，若计算时使用的间隔距离比实际更小，则会导致速度相对实际更慢，使撞线时间变长。
