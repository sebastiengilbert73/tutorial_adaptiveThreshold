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
import ast
import utilities

parser = argparse.ArgumentParser()
parser.add_argument('imageFilepath', help="The filepath of the test image")
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--blurSize', help="The blurring neighborhood size. Default: '(11, 11)'", default='(11, 11)')
parser.add_argument('--adaptiveThresholdNeighborhoodSide', help="The adaptive threshold neighborhood side. Default: 35", type=int, default=35)
parser.add_argument('--adaptiveThresholdCForBrightPixels', help="The adaptive threshold C parameter for bright pixels. Default: -7", type=int, default=-7)
parser.add_argument('--adaptiveThresholdCForDarkPixels', help="The adaptive threshold C parameter for dark pixels. Default: 7", type=int, default=7)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')
blurSize = ast.literal_eval(args.blurSize)

def main():
    logging.info("adaptive_thresholding.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Load the test image
    testImg = cv2.imread(args.imageFilepath, cv2.IMREAD_GRAYSCALE)
    image_shapeHW = testImg.shape
    blurredImg = cv2.blur(testImg, ksize=blurSize)
    cv2.imwrite(os.path.join(args.outputDirectory, "adaptiveThresholding_original.png"), testImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "adaptiveThresholding_blurred.png"), blurredImg)

    # Apply adaptive threshold to detect bright areas
    adaptive_thresholded_img = cv2.adaptiveThreshold(src=blurredImg,
                                                     maxValue=255,
                                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     thresholdType=cv2.THRESH_BINARY,
                                                     blockSize=args.adaptiveThresholdNeighborhoodSide,
                                                     C=args.adaptiveThresholdCForBrightPixels)
    # Apply adaptive inverse threshold to detect dark areas
    adaptive_inverse_thresholded_img = cv2.adaptiveThreshold(src=blurredImg,
                                                             maxValue=255,
                                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             thresholdType=cv2.THRESH_BINARY_INV,
                                                             blockSize=args.adaptiveThresholdNeighborhoodSide,
                                                             C=args.adaptiveThresholdCForDarkPixels)
    cv2.imwrite(os.path.join(args.outputDirectory, "adaptiveThresholding_thresholded.png"), adaptive_thresholded_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "adaptiveThresholding_inverseThresholded.png"), adaptive_inverse_thresholded_img)

    anomalies_img = utilities.ColorizeAnomalies(adaptive_inverse_thresholded_img, adaptive_thresholded_img)
    original_anomalies_stack_img = utilities.StackColorImages([cv2.cvtColor(testImg, cv2.COLOR_GRAY2BGR), anomalies_img])
    cv2.imwrite(os.path.join(args.outputDirectory, "adaptiveThresholding_anomalies.png"), original_anomalies_stack_img)

if __name__ == '__main__':
    main()