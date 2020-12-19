# Video Inpainting

The full pipeline required to inpaint videos can be found in `inpainting_integration.py`.

There are four classes present in the file used in the pipeline:

1. `PedestrianDetector()`; the basis of the human detection algorithm that provides a masking matrix indicating target regions to be inpainted
2. `InpaintAlgorithm()`; the basis of the inpainting algorithm that inpaints target regions given an image
3. `InpaintVideo()`; utilizes InpaintAlgorithm() to inpaint a given video
4. `Inpainter()`; integrates `PedestrianDetector()`, `InpaintAlgorithm()` and `InpaintVideo()` to provide a video processing pipeline

## Quick Start

To run inpainting on a given video:

```
pip install -r requirements.txt
python inpainting_integration.py -f <path-to-video>
```

The script will output a video file in the current working directory.
