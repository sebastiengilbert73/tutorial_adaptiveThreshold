"""
Code by Sébastien Gilbert

Reference for the dataset:
https://www.aitex.es/afid/
AFID: a  public fabric image database for defect detection.
Javier Silvestre-Blanes, Teresa Albero-Albero, Ignacio Miralles, Rubén Pérez-Llorens, Jorge Moreno
AUTEX Research Journal, No. 4, 2019

Note: Mask_images/0044_019_04_mask1.png and
                  0044_019_04_mask2.png
                ... have been merged into
                  0044_019_04_mask.png
      Mask_images/0097_030_03_mask1.png and
                  0097_030_03_mask2.png
                ... have been merged into
                  0097_030_03_mask.png
      Mask_images/0100_025_08_mask.png was created manually since it was missing in the original dataset
"""

import logging
import argparse
import os
import cv2
import numpy as np
import ast
import utilities

parser = argparse.ArgumentParser()
parser.add_argument('imageFilepath', help="The filepath of the test image")
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--blurSize', help="The blurring neighborhood size. Default: '(11, 11)'", default='(11, 11)')
parser.add_argument('--thresholdDeltaAboveMedian', help="The additive value to the median gray level for the bright pixels threshold. Default: 12", type=int, default=12)
parser.add_argument('--thresholdDeltaBelowMedian', help="The subtractive value to the median gray level for the dark pixels inverse threshold. Default: 12", type=int, default=12)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')
blurSize = ast.literal_eval(args.blurSize)

def main():
    logging.info("uniform_thresholding.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Load the test image
    testImg = cv2.imread(args.imageFilepath, cv2.IMREAD_GRAYSCALE)
    image_shapeHW = testImg.shape
    blurredImg = cv2.blur(testImg, ksize=blurSize)
    cv2.imwrite(os.path.join(args.outputDirectory, "uniformThresholding_original.png"), testImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "uniformThresholding_blurred.png"), blurredImg)

    # Compute the image median
    image_median = np.median(blurredImg)
    logging.info("image_median = {}".format(image_median))

    # Apply threshold to detect bright areas
    _, thresholded_img = cv2.threshold(src=blurredImg,
                                       thresh=image_median + args.thresholdDeltaAboveMedian,
                                       maxval=255,
                                       type=cv2.THRESH_BINARY)
    # Apply inverse threshold to detect dark areas
    _, inverse_thresholded_img = cv2.threshold(blurredImg,
                                               thresh=image_median - args.thresholdDeltaBelowMedian,
                                               maxval=255,
                                               type=cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(args.outputDirectory, "uniformThresholding_thresholded.png"), thresholded_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "uniformThresholding_inverseThresholded.png"), inverse_thresholded_img)

    anomalies_img = utilities.ColorizeAnomalies(inverse_thresholded_img, thresholded_img)
    original_anomalies_stack_img = utilities.StackColorImages([cv2.cvtColor(testImg, cv2.COLOR_GRAY2BGR), anomalies_img])
    cv2.imwrite(os.path.join(args.outputDirectory, "uniformThresholding_anomalies.png"), original_anomalies_stack_img)

if __name__ == '__main__':
    main()