import numpy as np

def ColorizeAnomalies(dark_anomalies_img, bright_anomalies_img):
    if dark_anomalies_img.shape != bright_anomalies_img.shape:
        raise ValueError("utilities.ColorizeAnomalies(): dark_anomalies_img.shape ({}) != bright_anomalies_img.shape ({})".format(
            dark_anomalies_img.shape, bright_anomalies_img.shape ))
    img_shapeHW = dark_anomalies_img.shape
    colorized_img = np.zeros((img_shapeHW[0], img_shapeHW[1], 3), dtype=np.uint8)
    colorized_img[:, :, 0] = dark_anomalies_img  # blue
    colorized_img[:, :, 2] = bright_anomalies_img  # red
    return colorized_img

def StackColorImages(images_list):
    number_of_images = len(images_list)
    img_shapeHW = (images_list[0].shape[0], images_list[0].shape[1])
    stacked_img_shapeHW = (number_of_images * img_shapeHW[0], img_shapeHW[1])
    stacked_img = np.zeros((stacked_img_shapeHW[0], stacked_img_shapeHW[1], 3), dtype=np.uint8)
    for imgNdx in range(number_of_images):
        stacked_img[imgNdx * img_shapeHW[0]: (imgNdx + 1) * img_shapeHW[0], :, :] = images_list[imgNdx]
    return stacked_img