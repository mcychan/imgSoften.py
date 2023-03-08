# What is the imgSoften.py?


## Introduction


imgSoften.py is a collection of image Filters for smoothing and edge enhancement that include:
- Anisotropic Diffusion
- Bilateral
- EJBilateral
- FGS
- LPGPCA
- LS


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
