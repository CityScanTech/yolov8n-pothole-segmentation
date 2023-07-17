#! ~usr/bin/bash 
# train
yolo segment train data=./data_configs/pothole.yaml model="C:\Users\dongd\Desktop\yolov8n-pothole-segmentation\runs\segment\train14\weights\best.pt" epochs=100 batch=16 imgsz=640
# # val
# yolo segment val data=roadvis.yaml model=runs/segment/train17/weights/best.pt