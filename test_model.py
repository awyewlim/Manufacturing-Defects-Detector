import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('model/model1.h5')

image = cv2.imread('casting_data/test/def_front/cast_def_0_15.jpeg',0)
image = image/255
pred_img =image.copy()

prediction = model.predict(image.reshape(-1,300,300,1))
if (prediction<0.5):
    print("def_front")
    cv2.putText(pred_img, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("ok_front")
    cv2.putText(pred_img, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
plt.imshow(pred_img,cmap='gray')
plt.axis('off')
plt.show()


