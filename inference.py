#! ~/bin/python3

import glob
import os
import subprocess
from ultralyticsplus import render_result
from ultralytics import YOLO
import cv2
import numpy as np
from psd_tools import PSDImage

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # pre-set layer names in the PSD files
    # layer_names = ['pothole', 'patch']
    # example: 
    layer_names = [
        'transverse', 'transverse_seal_good', 'transverse_seal_poor', 
        'longitudinal', 'longitudinal_seal_good', 'longitudinal_seal_poor', 
        'alligator', 'alligator_seal_good', 'alligator_seal_poor', 
        'block', 'block_seal_good', 'block_seal_poor', 
        'slippage', 
        'pothole', 'patch', 'patch_poor'
    ]

    # load model
    model = YOLO(r"runs\segment\train3\weights\best.pt")

    # set model parameters
    model.overrides['conf'] = 0.5  # NMS confidence threshold
    model.overrides['iou'] = 0.5  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 10 # maximum number of detections per image

    # load images
    filePath = 'dataset/inference'
    resultPath = filePath+'_result'

    # Get a list of all files in the directory
    file_list = glob.glob(os.path.join(filePath, '*'))

    # Loop through each file
    for image_path in file_list:
        
        file_name = os.path.basename(image_path)
        image_name = file_name.split('_')[0]
        print("Predict result for image: " + image_name)
        mask_folder = os.path.join(resultPath, image_name)
        os.makedirs(mask_folder, exist_ok=True)

        # perform inference
        results = model.predict(image_path)
        result = results[0]
        # save masks as png files
        h, w = result.orig_shape
        blank_rgb = np.zeros((h,w,3))
        alpha_channel = np.zeros((h,w)) 
        blank_image = cv2.merge((blank_rgb, alpha_channel))
        if not result.masks: 
            for name in layer_names:
                cv2.imwrite(os.path.join(mask_folder, '{}.png'.format(name)), blank_image) 
        else:
            merge_mask = {k:[] for k in layer_names}
            for i in range(len(result.masks)): # get all predicted instances
                mask = result.masks.data[i,:,:] 
                mask = mask.cpu().numpy() # To convert to numpy
                mask = cv2.resize(mask, (w,h))
                mask[mask>0] = 255 # hard threshold
                alpha = mask.copy() # construct alpha layer (for transparency)
                mask = mask[np.newaxis, :]
                mask = np.transpose(mask, (1, 2, 0))
                label = int(result.boxes.cls[i].item())
                category = result.names[label]
                merge_mask[category].append(mask)
                mask = 255-mask # revert black/white
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 1 channel to 3 channels
                mask = cv2.merge((mask, alpha)) # add alpha channel (4 channels in total)
                cv2.imwrite(os.path.join(
                    mask_folder, '{}_{}.png'.format(category,len(merge_mask[category]))
                ), mask)
            for k, v in merge_mask.items(): # combine all instances of each category 
                if not v: 
                    cv2.imwrite(os.path.join(mask_folder, '{}.png'.format(k)), blank_image) 
                    continue
                mask_k = np.sum(np.stack(v, axis=0), axis=0)
                mask_k[mask_k>0] = 255
                alpha_k = mask_k.copy()
                mask_k = 255-mask_k
                mask_k = cv2.cvtColor(mask_k, cv2.COLOR_GRAY2BGR)
                mask_k = cv2.merge((mask_k, alpha_k))
                cv2.imwrite(os.path.join(mask_folder, '{}.png'.format(k)), mask_k)

        # cli: ImageMagick to merge masks into a psd file
        psd_components = [image_path,image_path]
        for name in layer_names:
            mask_path = os.path.join(mask_folder, name+'.png')
            if os.path.isfile(mask_path):
                psd_components.append(mask_path)
        cmd = [
            "magick", 
        #     "-background", "white",
        #     "-gravity", "center",
        ]
        output_path = [os.path.join(resultPath, '{}.psd'.format(image_name))]
        cmd = cmd + psd_components + output_path
        p = subprocess.run(cmd, capture_output=True)
        # print(p)

        # change layers names in psd files
        psd = PSDImage.open(output_path[0])
        for i, layer in enumerate(psd):
            if i==0: 
                layer.name = 'overhead' 
            else: 
                layer.name = layer_names[i-1]
        psd.save(output_path[0])

        # observe results
        render = render_result(model=model, image=image_path, result=result)
        render.save(os.path.join(resultPath, '{}.png'.format(image_name)))
        # render.show()

        # save label txt
        result.save_txt(os.path.join(resultPath, '{}.txt'.format(file_name)))
