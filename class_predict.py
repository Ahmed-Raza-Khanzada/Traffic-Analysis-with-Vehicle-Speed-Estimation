from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import random
import string
import os
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)



def predict_class_mymodel(image,class_names, model):
   
  

    id_length = 7
    if image is None:
        return "no class"
    # if image.shape[0]<20 or image.shape[1]<20:
    #     return "no class"
    # Generate a random ID using uppercase letters and digits
    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=id_length))
    while random_id  in os.listdir('H:/upwork/vehicle-submission/vehicle-speed-estimation-v7/output_images_classification/arrange_data'):
       random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=id_length))
    # cv2.imwrite( f'H:/upwork/vehicle-submission/vehicle-speed-estimation-v7/output_images_classification/arrange_data/{random_id}.jpg',image)

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    # cv2.imshow("Webcam Image", image)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:],f"Conf {confidence_score}", end="")
    if class_name[2:-1]=="other" and confidence_score<0.9:
        return "Bus"   

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    return class_name[2:-1]