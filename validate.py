# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import glob
import os
from ultralytics import YOLO
import wandb

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # load model
    wandb.init(mode='disabled')
    model = YOLO(r"runs\segment\train2\weights\best.pt")
    model.task='segment'

    # start validation
    results = model.val(
        data=r"data_configs\pothole.yaml", 
    )
    print('Metrics\tPothole\tPatch')
    print('Precisions:\t{}\t{}'.format(results.seg.p[0], results.seg.p[1]))
    print('Recall:\t{}\t{}'.format(results.seg.r[0], results.seg.r[1]))
    print('f1:\t{}\t{}'.format(results.seg.f1[0], results.seg.f1[1]))

