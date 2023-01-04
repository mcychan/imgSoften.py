import cv2
import numpy as np
import pathlib
import sys
import traceback

from LSFilter import LSFilter

def main(file_name):
    target_file = str(pathlib.Path().absolute()) + file_name
    # Reading an image in default mode
    src = cv2.imread(target_file)
    dst = LSFilter(src).filter()

    htich = np.hstack((src, dst))
    cv2.imshow('merged_img', htich)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file_name = "/samples/peppers.jpg"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    try:
        main(file_name)
    except:
        traceback.print_exc()
