# What is the imgSoften.py?


## Introduction


imgSoften.py is a collection of image Filters for smoothing and edge enhancement that include:
- Anisotropic Diffusion
- Bilateral Filter
- Extended Joint Bilateral Filter
- Fast Global Image Smoothing based on Weighted Least Squares
- Local Pixel Grouping Pattern Recognition Filter
- Least-squares images Filter


```python
import cv2
import sys

from pathlib import Path
from Evaluate import *
from EJBilateralFilter import EJBilateralFilter
```

```python
# Reading an image in default mode
src = cv2.imread(target_file)
```

```python
dst = EJBilateralFilter(src, 5, 4).filter()
```
