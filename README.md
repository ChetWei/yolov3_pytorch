# yolov3_pytorch
yolov3 implement of pytorch

## 1.数据集制作
### 1.1 VOC 标签文件制作
- （1）先固定数据集的类别名称文件
- （2）将所有标签读取并写入到一个文件中
```
图片1路径  clsid,x1,y1,x2,y2 clsid,x1,y1,x2,y2
图片2路径  clsid,x1,y1,x2,y2 clsid,x1,y1,x2,y2
...
```
这里的xy都是使用绝对像素点

### 1.2 dataset 类
 - 用PIL库读取图片，转换为RGB模式
 - 填充图片为正方形，边缘填充，要调整中心点的位置
 - 将图片resize成输入的尺寸，同时调整box
 - 转换图片为 （3,input_h,input_w）的维度值
 - 返回图片 和标签数据

## 2.创建模型
模型Head的输出
```
torch.Size([1, 75, 13, 13])
torch.Size([1, 75, 26, 26])
torch.Size([1, 75, 52, 52])
```
对模型输出进行预处理，并且添加一些信息


## 3.设计损失函数

## 4.预测结果后处理 NMS

## 5.测试map

## 5.图片检测、视频检测