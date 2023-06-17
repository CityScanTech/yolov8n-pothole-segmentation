#! ~usr/bin/bash 
# train
yolo segment train data=./data_configs/roadvis.yaml model=yolov8m-seg.pt epochs=100 batch=16 imgsz=640
# # val
# yolo segment val data=roadvis.yaml model=runs/segment/train17/weights/best.pt