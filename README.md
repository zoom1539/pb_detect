# pb_detect
## platform basic algorithm: detect
- refer to https://github.com/wang-xinyu/tensorrtx.git
- yolov5s.engine in google drive(model_zoo/pb_detect)
## IMPORTANT
- for different class_num:
1) change CLASS_NUM in yololayer.h 
```
static constexpr int CLASS_NUM = 80;
```
2) then make a lib.so with a special class_num to use
```
make install
```
