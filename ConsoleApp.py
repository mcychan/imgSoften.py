import cv2
from pathlib import Path
import sys
import traceback

from Evaluate import *

from AnisotropicDiffusionFilter import AnisotropicDiffusionFilter
from BilateralFilter import BilateralFilter
from EJBilateralFilter import EJBilateralFilter
from FGSFilter import FGSFilter
from LPGPCAFilter import LPGPCAFilter
from LSFilter import LSFilter

def main(file_name, filter_name):
    target_file = file_name if Path(file_name).exists() else str(Path().absolute()) + file_name
    # Reading an image in default mode
    src = cv2.imread(target_file)

    match filter_name:
        case "AnisotropicDiffusion":
            dst = AnisotropicDiffusionFilter(src).filter()
        case "Bilateral":
            dst = BilateralFilter(src).filter()
        case "FGS":
            dst = FGSFilter(src).filter()
        case "LPGPCA":
            dst = LPGPCAFilter(src).filter()
        case "LS":
            dst = LSFilter(src).filter()
        case _:
            filter_name = "EJBilateral"
            dst = EJBilateralFilter(src, 5, 4).filter()

    psnr, ssim = calculate_psnr(src, dst), calculate_ssim(src, dst)
    print("Filter:", filter_name, "PSNR:", psnr, "SSIM:", ssim)
    htich = np.hstack((src, dst))
    cv2.imshow('merged_img', htich)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file_name, filter_name = "/samples/beach.jpg", ""
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        filter_name = sys.argv[2]

    try:
        main(file_name, filter_name)
    except:
        traceback.print_exc()
