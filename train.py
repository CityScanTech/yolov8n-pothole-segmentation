# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ultralytics import YOLO
import wandb

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # load model
    wandb.init(mode='disabled')
    model = YOLO(r"yolov8m-seg.pt")
    model.task='segment'
    # start training
    model.train(
        batch=16,
        device="0",
        data=r"data_configs\pothole.yaml", 
        epochs=100,
        imgsz=640,
        pretrained=True, 
        val=False,
    )

    val_results = model.val()
