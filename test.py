from PIL import Image
from ultralytics import YOLO

model = YOLO('weight/yolo_cable.pt')

# Run live inference from webcam
#results = model(source=0, show=True, conf=0.4)

results = model('cable.jpg', conf=0.4)

for r in results:
    im_array = r.plot(
        boxes=False,   
        masks=True
    )  
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('results.jpg')