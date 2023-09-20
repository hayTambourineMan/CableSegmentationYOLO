# CableSegmentationYOLO
Cable Segmentation using a YOLO model

## Install

Need to install the ultralytics package in your environment: `pip install ultralytics`

## Input Source

Depending on the types of input source (cached data structure), can pass in:
- Single image file: `image.jpg` as str or Path
- PIL image: `Image.open("im.jpg")` as PIL.Image
- OpenCV image: `cv2.imread("im.jpg")` as np.ndarray
- numpy array: `np.zeros((640, 640, 3))` as np.ndarray
- torch tensor: `torch.zeros(16, 3, 320, 640)` as torch.Tensor
- Can pass in various image format: `.bmp`, `.dng`, `.jpeg`, `.jpg`, `.mpo`, `.png`, `.tif`, `.tiff`, `.webp`, `.pfm`.

## Run Inference

To run the inference on the YOLO model:
- First load the trained weight: `model = YOLO("yolo_cable.pt")`
- Then load the image or array: `source = InputSourceAbove`
- Lastly run the inference on the source: `result = model(source)`
- If only want mask, then call `plot(boxes=False, masks=True)` on the `Result` object.