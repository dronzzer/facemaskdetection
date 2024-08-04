import cv2
import numpy as np
from keras.models import load_model
from config import LABEL_DICT


def detect_and_classify(model, label_dict, img_size=100):
    # Ensure colors correspond to labels
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}  # Example: Green for No Mask, Red for Mask
    labels_dict = {key: value for key, value in label_dict.items()}  # Ensure correct mapping

    face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    source = cv2.VideoCapture(0)

    while True:
        ret, img = source.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (img_size, img_size))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

            # Ensure model.predict is successful
            try:
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]  # Ensure label is assigned before usage

                if label not in labels_dict:
                    print(f"Label {label} not found in labels_dict!")  # Debug: Missing label
                    continue  # Skip labels not found

                print("Predicted label index:", label)  # Debug: Check predicted label

                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict.get(label, (255, 0, 0)), 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict.get(label, (255, 0, 0)), -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error during prediction: {e}")  # <-- Catch any errors during prediction

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
    source.release()


if __name__ == "__main__":
    model = load_model('model.keras')
    detect_and_classify(model, LABEL_DICT)
