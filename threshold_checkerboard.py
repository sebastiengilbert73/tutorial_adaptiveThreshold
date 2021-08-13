import logging
import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath of the test image. Default: './images/lighting_checkerboard.png'", default='./images/lighting_checkerboard.png')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--flatThreshold', help="Threshold value for the flat thresholding. Default: 127", type=int, default=127)
parser.add_argument('--neighborhoodSide', help="The side of the neighborhood for the adaptive thresholding. Default: 171", type=int, default=171)
parser.add_argument('--adaptiveThresholdC', help="The C parameter, for the adaptive thresholding. Default: -18", type=int, default=-18)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("threshold_checkerboard.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Load the test image
    testImg = cv2.imread(args.imageFilepath, cv2.IMREAD_GRAYSCALE)

    # Flat thresholding
    _, thresholded_img = cv2.threshold(src=testImg,
                                       thresh=args.flatThreshold,
                                       maxval=255,
                                       type=cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholdCheckerboard_flatThresholded.png"), thresholded_img)

    # Adaptive thresholding
    adaptive_thresholded_img = cv2.adaptiveThreshold(testImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY, args.neighborhoodSide,
                                                     args.adaptiveThresholdC)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholdCheckerboard_adaptiveThresholded.png"), adaptive_thresholded_img)

if __name__ == '__main__':
    main()