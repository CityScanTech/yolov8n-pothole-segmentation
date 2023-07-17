# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import glob
import os
from ultralyticsplus import render_result
from ultralytics import YOLO
import cv2
import numpy as np
from detectron2.structures.masks import BitMasks, PolygonMasks, polygons_to_bitmask

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # load model
    model = YOLO(r"runs\segment\train3\weights\best.pt")

    # set model parameters
    model.overrides['conf'] = 0.5  # NMS confidence threshold
    model.overrides['iou'] = 0.5  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 10 # maximum number of detections per image

    # load images
    filePath = 'dataset/inference'

    # Get a list of all files in the directory
    file_list = glob.glob(os.path.join(filePath, '*'))

    # Loop through each file
    for image in file_list:
        file_name = os.path.basename(image)
        print("Predict result for image: " + file_name)

        # perform inference
        results = model.predict(image)

        # save label txt
        results[0].save_txt(os.path.join('dataset/inference_result', file_name+'.txt'))
        for xy in results[0].masks.xy:
            mask = np.zeros((results[0].orig_shape))
            cv2.fillConvexPoly(mask, xy, 1)
            mask = mask > 0 # To convert to Boolean
            cv2.imshow('Extracted Image', mask)
            import pdb
            pdb.set_trace()




        # observe results
        # print(results[0].boxes)
        # print(results[0].masks)
        render = render_result(model=model, image=image, result=results[0])
        render.save('dataset/inference_result/' + file_name, 'png')
        # render.show()
        import pdb
        pdb.set_trace()
