import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
#import tensorflow as tf

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image - 255) / 255, axis=0)

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    results = [
        {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i]
        } for i in range(count) if scores[i] >= threshold
    ]

    return results

def process_image(img, object_detection_interpreter, character_recognition_interpreter, threshold):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    results = detect_objects(object_detection_interpreter, img, threshold)
    #print(results)
    if results:
        # Crop the last detected license plate without drawing anything
        last_result = results[-1]
        #print(last_result)
        ymin, xmin, ymax, xmax = last_result['bounding_box']
        xmin = int(max(1, xmin * img.shape[1]))
        xmax = int(min(img.shape[1], xmax * img.shape[1]))
        ymin = int(max(1, ymin * img.shape[0]))
        ymax = int(min(img.shape[0], ymax * img.shape[0]))

        # Crop the license plate region
        cropped_plate = img[ymin:ymax, xmin:xmax]

        # Uncomment the following line to display characters using the recognition model
        out = pred_characters(cropped_plate, character_recognition_interpreter)
        return out
def pred_characters(image, interpreter):
    classes_name = os.listdir("C:/Users/mhhas/Documents/Licsence/iranis-datasets")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better handling of lighting variations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply erosion to reduce noise
    kernel_erosion = np.ones((1, 1), np.uint8)
    thresh = cv2.erode(thresh, kernel_erosion, iterations=1)

    # Apply opening to smooth contours
    kernel_opening = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    characters = []

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if 350 > area > 50:
            x, y, w, h = cv2.boundingRect(c)

            # Further isolate individual characters using horizontal projection or other techniques
            ROI = gray[y:y + h, x:x + w]
            characters.append((i + 1, ROI))

    # Load the TensorFlow Lite model
    interpreter.allocate_tensors()

    # Display characters in separate matplotlib subplots
    if characters:
        output = ''
        for character_index, character in characters:
            print(f"Character {character_index} shape:", character.shape)
            
            # Check if the character shape is as expected
            if character.shape[0] > 0 and character.shape[1] > 0:
                img = cv2.resize(character, (64, 64))
                img = img / 255.0  # Normalize pixel values to be between 0 and 1
                img = np.expand_dims(img, axis=(0, -1))
                img = img.astype(np.float32)
                input_tensor_index = interpreter.get_input_details()[0]['index']
                interpreter.set_tensor(input_tensor_index, img)
                interpreter.invoke()
                output_tensor_index = interpreter.get_output_details()[0]['index']
                predictions = interpreter.get_tensor(output_tensor_index)
                predicted_label = np.argmax(predictions, axis=1)[0]

                output += classes_name[predicted_label]
                
            else:
                print(f"Skipping character {character_index} due to invalid shape")
    print("out is", output)
    return output


def process(image):
    # Load the object detection model
    object_detection_interpreter = Interpreter('detect.tflite')
    object_detection_interpreter.allocate_tensors()

    # Load the character recognition model
    character_recognition_interpreter = Interpreter('Plate_1Dec.tflite')
    character_recognition_interpreter.allocate_tensors()

    threshold = 0.6

    output = process_image(image, object_detection_interpreter, character_recognition_interpreter, threshold)
    print("Process output: ", output)
    return output
app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_route():
    # Receive image from the request
    img_data = request.files['image'].read()
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image
    result = process(image)
    print("Result is :",result)
    # Return the result as JSON
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
