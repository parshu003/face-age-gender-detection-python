import cv2
from deepface import DeepFace

# Load image
img = cv2.imread("third.jpg")

if img is None:
    print("Image not found")
    exit()

results = DeepFace.analyze(
    img_path=img,
    actions=['age', 'gender'],
    enforce_detection=False
)

if not isinstance(results, list):
    results = [results]

for res in results:
    region = res['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    age = res['age']
    gender = res['dominant_gender']

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{gender}, {age}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

img_resized = cv2.resize(img, (1200, 800))
cv2.imshow("Age & Gender Detection", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("END OF THE PROGRAM!")
