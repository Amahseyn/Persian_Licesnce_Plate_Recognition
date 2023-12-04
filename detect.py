import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)

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

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def process_image(image_path, interpreter, threshold):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    results = detect_objects(interpreter, img, threshold)

    if results:
        # Crop the last detected license plate without drawing anything
        last_result = results[-1]
        ymin, xmin, ymax, xmax = last_result['bounding_box']
        xmin = int(max(1, xmin * img.shape[1]))
        xmax = int(min(img.shape[1], xmax * img.shape[1]))
        ymin = int(max(1, ymin * img.shape[0]))
        ymax = int(min(img.shape[0], ymax * img.shape[0]))
        
        # Crop the license plate region
        cropped_plate = img[ymin:ymax, xmin:xmax]

        # Save the cropped plate or further process as needed
        cv2.imshow('Cropped License Plate', cropped_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()

    image_path = '1.jpg'  # Specify the path to your image
    threshold = 0.6

    process_image(image_path, interpreter, threshold)

if __name__ == "__main__":
    main()
