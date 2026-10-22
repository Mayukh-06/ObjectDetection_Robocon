import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
  print("Error: Could not open webcam.")
  exit()

print("Running webcam detection... Press 'q' to exit.")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    print("Error: Failed to grab frame.")
    break

  results = model(frame, stream=True, conf=0.85)

  for r in results:
    for box in r.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      conf = float(box.conf[0])

      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

      cv2.putText(
          frame,
          f"({x1},{y1}) conf: {conf:.2f}",
          (x1, max(y1 - 10, 15)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

      cv2.putText(
          frame,
          f"({x2},{y2})",
          (x2-100, y2 + 20),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )


  cv2.imshow("Object Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()

cv2.destroyAllWindows()