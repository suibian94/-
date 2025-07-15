from ultralytics import YOLO
import cv2
# Load a COCO-pretrained YOLOv8n model
model = YOLO("./model/yolov8n.pt")

# Display model information (optional)
#model.info()

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
img = cv2.imread("./AA.jpg")
results = model("./AA.jpg")
print(results)
print("="*100)
for results in results[0].boxes:
    id=results.cpu().cls.squeeze().item()
    print(id)
x1,x2,y1,y2=results.cpu().xyxy[0].numpy().astype(int)
label="id={:d}  num={:.2f}".format(int(id),results.cpu().conf.squeeze().item())
cv2.putText(img,label,(x2,y2-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.imshow("results",img)
cv2.waitKey(0)