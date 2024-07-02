import cv2
import os
import numpy as np
import pickle
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

def zhang_suen_thinning(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    ret, img_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_bin = eroded.copy()
        zeros = size - cv2.countNonZero(img_bin)
        if zeros == size:
            done = True
    return skel

def load_model():
    model, template_files, template_dir = None, None, None
    try:
        with open('template_matching_model.pickle', 'rb') as f:
            model = pickle.load(f)
            template_files = model['template_files']
            template_dir = model['template_dir']
            print("Model loaded successfully.")
            print("Model structure:", model)
    except Exception as e:
        print("Failed to load model:", str(e))
    return model, template_files, template_dir

model, template_files, template_dir = load_model()

def save_last_image_count(last_image_count):
    with open('last_image_count.txt', 'w') as f:
        for key, value in last_image_count.items():
            if isinstance(value, dict) and 'upper' in value and 'lower' in value:
                f.write(f'{key}_upper:{str(value["upper"])}\n')
                f.write(f'{key}_lower:{str(value["lower"])}\n')
            else:
                print(f"Invalid format for key '{key}': {value}")

def load_last_image_count():
    last_image_count = {}
    if os.path.exists('last_image_count.txt'):
        with open('last_image_count.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key, value = parts
                    parts_key = key.split('_')
                    if len(parts_key) == 2:
                        char, case = parts_key
                        if char not in last_image_count:
                            last_image_count[char] = {"upper": 0, "lower": 0}
                        last_image_count[char][case] = int(value)
    return last_image_count

def save_template_counts(template_correct_detections, template_total_counts):
    with open('template_counts.txt', 'w') as f:
        for template, count in template_total_counts.items():
            detections = template_correct_detections.get(template, 0)
            file_type = "upper" if "upper" in template else "lower"
            f.write(f'{template}:{file_type}:{str(count)}:{str(detections)}\n')

def load_template_counts():
    template_correct_detections = {}
    template_total_counts = {}
    if os.path.exists('template_counts.txt'):
        with open('template_counts.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(':')
                if len(parts) == 4:
                    template, file_type, count, detections = parts
                    template_correct_detections[template] = int(detections)  # Perbaikan konkatenasi
                    template_total_counts[template] = int(count)  # Perbaikan konkatenasi
    return template_correct_detections, template_total_counts

def get_threshold(alphabet_char, file_type):
    threshold_similarity = {
        'a': {'upper': (0.29, 0.35), 'lower': (0.2, 0.27)},
        'b': {'upper': (0.2, 0.28), 'lower': (0.25, 0.29)},
        'c': {'upper': (0.24, 0.33), 'lower': (0.26, 0.31)},
        'd': {'upper': (0.24, 0.4), 'lower': (0.22, 0.38)},
        'e': {'upper': (0.32, 0.41), 'lower': (0.24, 0.34)},
        'f': {'upper': (0.42, 0.49), 'lower': (0.34, 0.43)},
        'g': {'upper': (0.2, 0.23), 'lower': (0.2, 0.25)},
        'h': {'upper': (0.41, 0.49), 'lower': (0.3, 0.46)},
        'i': {'upper': (0.24, 0.42), 'lower': (0.2, 0.27)},
        'j': {'upper': (0.3, 0.37), 'lower': (0.2, 0.31)},
        'k': {'upper': (0.28, 0.32), 'lower': (0.3, 0.37)},
        'l': {'upper': (0.37, 0.6), 'lower': (0.33, 0.51)},
        'm': {'upper': (0.26, 0.3), 'lower': (0.29, 0.43)},
        'n': {'upper': (0.22, 0.3), 'lower': (0.36, 0.56)},
        'o': {'upper': (0.21, 0.25), 'lower': (0.2, 0.24)},
        'p': {'upper': (0.27, 0.34), 'lower': (0.2, 0.25)},
        'q': {'upper': (0.22, 0.29), 'lower': (0.2, 0.24)},
        'r': {'upper': (0.24, 0.31), 'lower': (0.26, 0.42)},
        's': {'upper': (0.26, 0.33), 'lower': (0.23, 0.27)},
        't': {'upper': (0.39, 0.57), 'lower': (0.29, 0.41)},
        'u': {'upper': (0.25, 0.28), 'lower': (0.3, 0.41)},
        'v': {'upper': (0.39, 0.47), 'lower': (0.36, 0.43)},
        'w': {'upper': (0.28, 0.31), 'lower': (0.22, 0.25)},
        'x': {'upper': (0.24, 0.33), 'lower': (0.25, 0.31)},
        'y': {'upper': (0.3, 0.46), 'lower': (0.26, 0.34)},
        'z': {'upper': (0.3, 0.47), 'lower': (0.25, 0.38)}
    }
    return threshold_similarity.get(alphabet_char, {}).get(file_type, 0.0)

def get_similarity_threshold(alphabet_char, file_type):
    threshold_similarity = 0.0
    threshold_similarity = {
        'a': {'upper': (0.54, 0.58), 'lower': (0.3, 0.32)},
        'b': {'upper': (0.32, 0.36), 'lower': (0.54, 0.61)},
        'c': {'upper': (0.74, 0.78), 'lower': (0.26, 0.31)},
        'd': {'upper': (0.35, 0.39), 'lower': (0.71, 0.74)},
        'e': {'upper': (0.43, 0.5), 'lower': (0.26, 0.28)},
        'f': {'upper': (0.64, 0.74), 'lower': (0.47, 0.56)},
        'g': {'upper': (0.42, 0.46), 'lower': (0.27, 0.3)},
        'h': {'upper': (0.28, 0.36), 'lower': (0.51, 0.56)},
        'i': {'upper': (0.32, 0.4), 'lower': (0.34, 0.55)},
        'j': {'upper': (0.56, 0.65), 'lower': (0.51, 0.57)},
        'k': {'upper': (0.49, 0.53), 'lower': (0.62, 0.66)},
        'l': {'upper': (0.51, 0.6), 'lower': (0.52, 0.65)},
        'm': {'upper': (0.32, 0.37), 'lower': (0.37, 0.4)},
        'n': {'upper': (0.33, 0.35), 'lower': (0.3, 0.41)},
        'o': {'upper': (0.41, 0.44), 'lower': (0.36, 0.38)},
        'p': {'upper': (0.48, 0.54), 'lower': (0.35, 0.48)},
        'q': {'upper': (0.38, 0.4), 'lower': (0.43, 0.48)},
        'r': {'upper': (0.35, 0.41), 'lower': (0.55, 0.6)},
        's': {'upper': (0.34, 0.37), 'lower': (0.3, 0.41)},
        't': {'upper': (0.65, 0.77), 'lower': (0.53, 0.61)},
        'u': {'upper': (0.34, 0.4), 'lower': (0.36, 0.42)},
        'v': {'upper': (0.29, 0.35), 'lower': (0.27, 0.3)},
        'w': {'upper': (0.31, 0.35), 'lower': (0.31, 0.34)},
        'x': {'upper': (0.6, 0.64), 'lower': (0.5, 0.54)},
        'y': {'upper': (0.65, 0.69), 'lower': (0.35, 0.39)},
        'z': {'upper': (0.6, 0.7), 'lower': (0.62, 0.7)}
        }
    return threshold_similarity.get(alphabet_char, {}).get(file_type, 0.0)

def save_matching_results(results):
    try:
        with open('matching_results.txt', 'a') as f:
            for template_name, matches in results.items():
                for match in matches:
                    filename = match['filename']
                    file_type = "upper" if "upper" in template_name else "lower"
                    max_val = match['max_val']
                    similarity = match['similarity']
                    result = match['result']
                    f.write(f"Filename: {filename}, File Type: {file_type}, Max Val: {max_val:.6f}, Similarity: {similarity:.6f}, Result: {result}\n")
    except Exception as e:
        print("Failed to save matching results:", str(e))

def save_accuracy_statistics(template_correct_detections, template_total_counts):
    with open('accuracy_statistics.txt', 'w') as f:
        for template_key in sorted(template_correct_detections.keys()):
            detections = template_correct_detections.get(template_key, 0)
            count = template_total_counts.get(template_key, 0)
            accuracy = detections / count if count > 0 else 0
            file_type = "upper" if "upper" in template_key else "lower"
            f.write(f"Image: {template_key}, File Type: {file_type}, Total number of images: {str(count)}, Total Correct Detections: {str(detections)}, Accuracy: {accuracy:.2%}\n")
            print(f"Image: {template_key}, File Type: {file_type}, Total number of images: {count}, Total Correct Detections: {detections}, Accuracy: {accuracy:.2%}")

def is_random_scribble(image, threshold=5000):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding to create a binary image
    _, img_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Count non-black pixels
    non_black_pixels = np.sum(img_bin > 0)
    return non_black_pixels > threshold


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({"error": "No image data received"}), 400
            
            image_file = request.files['image']
            alphabet_char = request.args.get('alphabet_char', '').strip().lower()
            file_type = request.args.get('file_type', '').strip().lower()

            if not alphabet_char.isalpha() or not file_type:
                return jsonify({"error": "Parameter 'alphabet_char' or 'file_type' is missing or invalid"}), 400

            last_image_count = load_last_image_count()
            
            input_base_name = f'{alphabet_char}_{file_type}_bg'
            
            if alphabet_char not in last_image_count:
                last_image_count[alphabet_char] = {"upper": 0, "lower": 0}

            last_image_count[alphabet_char][file_type] += 1
            current_image_count = last_image_count[alphabet_char][file_type]
            filename = f'{input_base_name}_in_{current_image_count}.png'
            filepath = os.path.join('uploads', filename)
            
            image_data = image_file.read()
            image_data_decoded = base64.b64decode(image_data)
            nparr = np.frombuffer(image_data_decoded, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Failed to decode the image"}), 400

            cv2.imwrite(filepath, img)

            save_last_image_count(last_image_count)

            template_dir = 'templates'
            results = {}
            template_correct_detections, template_total_counts = load_template_counts()
            template_key = f'{alphabet_char}_{file_type}_bg'
            template_files = [template_key]

            if os.path.exists(os.path.join(template_dir, f'{alphabet_char}_{file_type}_bg.png')):
                template_path = os.path.join(template_dir, f'{alphabet_char}_{file_type}_bg.png')
            else:
                return jsonify({"error": f"No template found for '{alphabet_char}'"}), 400

            # Load and process the template
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_thresh = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            template_thinned = zhang_suen_thinning(template_thresh)

            # Process the uploaded image
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img_thinned = zhang_suen_thinning(img_thresh)
            
            method = cv2.TM_CCOEFF_NORMED
            result = cv2.matchTemplate(template_thinned, img_thinned, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            similarity = cv2.matchShapes(template_thinned, img_thinned, cv2.CONTOURS_MATCH_I1, 0.0)

            # Adjust thresholds based on empirical values
            threshold_max_val = get_threshold(alphabet_char, file_type)
            threshold_similarity = get_similarity_threshold(alphabet_char, file_type)

            # Check for random scribbles
            if is_random_scribble(img_thinned):
                match_result = {"filename": filename, "max_val": max_val, "similarity": similarity, "result": "not same"}
            elif (threshold_max_val[0] <= max_val <= threshold_max_val[1]) and (threshold_similarity[0] <= similarity <= threshold_similarity[1]):
                match_result = {"filename": filename, "max_val": max_val, "similarity": similarity, "result": "same"}
                template_correct_detections[template_key] = template_correct_detections.get(template_key, 0) + 1
            else:
                match_result = {"filename": filename, "max_val": max_val, "similarity": similarity, "result": "not same"}

            template_total_counts[template_key] = template_total_counts.get(template_key, 0) + 1

            if template_key in results:
                results[template_key].append(match_result)
            else:
                results[template_key] = [match_result]

            save_matching_results(results)
            save_template_counts(template_correct_detections, template_total_counts)
            save_accuracy_statistics(template_correct_detections, template_total_counts)

            return jsonify(match_result), 200
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return jsonify({"error": "An error occurred while processing the request"}), 500


@app.route('/')
def index():
    return "Welcome to Character Recognition Prediction", 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5001)