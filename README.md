# imgSoften.py
Image Filters for smoothing and edge enhancement

"Edge-preserving image smoothing, which smooths small details and preserves sharp edges.
The flexible weighted least-squares optimization framework achieves edge-preserving smoothing by minimizing gradients using edge-aware weights.
Edge-aware image smoothing methods have also been used for image and video abstraction."

Anisotropic diffusion is implemented by means of an approximation of the generalized diffusion equation: each new image in the family is computed by applying this equation to the previous image. It is an iterative process where a relatively simple set of computation are used to compute each successive image in the family and this process is continued until a desired degree of smoothing is obtained.

Welcome to comment and discover more legendary image filters.
