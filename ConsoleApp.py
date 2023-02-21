import cv2
import numpy as np
from pathlib import Path
import sys
import traceback

from Evaluate import *

from EJBilateralFilter import EJBilateralFilter

def main(file_name):
    target_file = file_name if Path(file_name).exists() else str(Path().absolute()) + file_name
    # Reading an image in default mode
    src = cv2.imread(target_file)
    dst = EJBilateralFilter(src).filter()

    psnr, ssim = calculate_psnr(src, dst), calculate_ssim(src, dst)
    print("PSNR:", psnr, "SSIM:", ssim)
    htich = np.hstack((src, dst))
    cv2.imshow('merged_img', htich)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file_name = "/samples/beach.jpg"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    try:
        main(file_name)
    except:
        traceback.print_exc()
