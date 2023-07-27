# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import glob
import os
import subprocess
from ultralyticsplus import render_result
from ultralytics import YOLO
import cv2
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # get label map
    label_dict = {0:'pothole', 1:'patch'}

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
        mask_folder = os.path.join(filePath+'_result', file_name.split('_')[0])
        os.makedirs(mask_folder, exist_ok=True)

        # perform inference
        results = model.predict(image)
        # save masks as png files
        h, w = results[0].orig_shape
        blank_rgb = np.zeros((h,w,3))
        alpha_channel = np.zeros((h,w)) 
        blank_image = cv2.merge((blank_rgb, alpha_channel))
        cv2.imwrite(os.path.join(mask_folder, 'bkg.png'), blank_image) 
        if not results[0].masks: 
            cv2.imwrite(os.path.join(mask_folder, 'pothole.png'), blank_image) 
            cv2.imwrite(os.path.join(mask_folder, 'patch.png'), blank_image) 
        else:
            merge_mask = {0:[],1:[]}
            for i in range(len(results[0].masks)):
                mask = results[0].masks.data[i,:,:] #.cpu().numpy()
                mask = mask.cpu().numpy() # To convert to Boolean
                mask = cv2.resize(mask, (w,h))
                mask[mask>0] = 255
                alpha = mask.copy()
                mask = mask[np.newaxis, :]
                mask = np.transpose(mask, (1, 2, 0))
                label = int(results[0].boxes.cls[i].item())
                category = label_dict[label]
                merge_mask[label].append(mask)
                mask = 255-mask
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = cv2.merge((mask, alpha))
                cv2.imwrite(os.path.join(mask_folder, '{}_{}.png'.format(category,i)), mask)
            for k, v in merge_mask.items():
                if not v: 
                    cv2.imwrite(os.path.join(mask_folder, '{}.png'.format(label_dict[k])), blank_image) 
                    continue
                mask_k = np.sum(np.stack(v, axis=0), axis=0)
                mask_k[mask_k>0] = 255
                alpha_k = mask_k.copy()
                mask_k = 255-mask_k
                mask_k = cv2.cvtColor(mask_k, cv2.COLOR_GRAY2BGR)
                mask_k = cv2.merge((mask_k, alpha_k))
                # import pdb
                # pdb.set_trace()
                # mask_k = cv2.resize(mask_k, (w,h))
                cv2.imwrite(os.path.join(mask_folder, '{}.png'.format(label_dict[k])), mask_k)

        # cli: ImageMagick to merge masks into a psd file
        psd_components = [
            image,
            image,
        ]
        for c in ['pothole', 'patch']: 
            c_path = os.path.join(mask_folder, c+'.png')
            if os.path.isfile(c_path):
                psd_components.append(c_path)
        cmd = [
            "magick", 
        #     "-background", "white",
        #     "-gravity", "center",
        ]
        output_path = [os.path.join('./dataset/inference_result', '{}.psd'.format(file_name.split('_')[0]))]
        cmd = cmd + psd_components + output_path

        p = subprocess.run(cmd, capture_output=True)
        print(p)

        # TODO: change layers names in psd files


        # observe results
        # print(results[0].boxes)
        # print(results[0].masks)
        render = render_result(model=model, image=image, result=results[0])
        render.save('dataset/inference_result/' + file_name, 'png')
        # render.show()

        # save label txt
        results[0].save_txt(os.path.join('dataset/inference_result', file_name+'.txt'))
