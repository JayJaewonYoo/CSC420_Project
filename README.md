# Video Inpainting

The full pipeline required to inpaint videos can be found in `inpainting_integration.py`.

There are four main classes:

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

## Notebook

Alternatively, the script is provided in a notebook format that can be run using Google Colab. Simply upload `inpainting_integration.ipynb` to Google Colab or a Jupyter Notebook context and run the notebook cells sequentially. Prior to running the notebook in Google Colab, the runtime type must be set to "GPU", and "pedestrians.avi" found in the submitted zip file on Quercus must be uploaded to the Google Colab directory.

## Results

Results of the inpainting algorithm on images can be found under the `\gifs` folder. The gif demonstrates the step by step progress of the algorithm inpainting the masked region. Additional outputs of the inpainting algorithm used in our report can be found in the `results` folder. 
