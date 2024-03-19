import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def image2sequence(img):
    """
    Convert an 256x256 array into a translated sequencel.
    return the sequence and the confidence.
    """
    left_tmp = np.load('left.npy')
    right_tmp = np.load('right.npy')
    result = ""

    # Normalize into [0,1]

    img -= img.min()
    img = img / img.max()

    conf = 1.

    for i in range(16):
        crop = img[116:147, 33+i*12:43+i*12]

        x, L, R = 1-crop.reshape(-1), 1-left_tmp.reshape(-1), 1-right_tmp.reshape(-1)
        sim_L = x.dot(L) / (np.linalg.norm(x) * np.linalg.norm(L))
        sim_R = x.dot(R) / (np.linalg.norm(x) * np.linalg.norm(R))

        if sim_L > sim_R + 0.5:
            result += '('
        elif sim_R > sim_L + 0.5:
            result += ')'
        else:
            result += '?'

        conf *= max(sim_L, sim_R)

    return result, conf

def isValid(expression: str) -> bool:
    stack = []
    bracket_pairs = {
        ')': '(',
        '}': '{',
        ']': '['
    }

    for char in expression:
        # If the character is an opening bracket, push it onto the stack.
        if char in bracket_pairs.values():
            stack.append(char)
        # If the character is a closing bracket, check if the top of the stack
        # has its corresponding opening bracket.
        elif char in bracket_pairs.keys():
            if stack == [] or bracket_pairs[char] != stack.pop():
                return False
        else:
            return False

    # The expression is valid only if the stack is empty at the end.
    return stack == []


def preprocess_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, 0)

    # Convert to binary using Otsu's thresholding
    _, binary = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    return binary


def extract_regions(binary_image):
    # Extract only the predetermined y-axis range
    binary_image = binary_image[116:147, :]

    # Project onto the x-axis
    projection = np.sum(binary_image, axis=0)

    # Find segments (where projection > 0)
    start = None
    regions = []
    for i in range(len(projection)):
        if projection[i] > 0 and start is None:
            start = i
        elif projection[i] == 0 and start is not None:
            end = i
            regions.append((start, 116, end - start, 31))
            #print(f"{start}:{end}, len {end - start if start else end}")
            start = None

    return regions


def match_template(region, templates, threshold=0.25):
    best_match = None
    best_score = -float('inf')

    for char, template in templates.items():
        # Resize the region to match template size
        resized = cv2.resize(region, (template.shape[1], template.shape[0]))
        # Compute the similarity
        score = np.mean(resized/255 * template/255)
        #print(f"{char}: prod is {score}")
        if score > best_score:
            best_score = score
            best_match = char

    #print(best_score)
    # print(region)
    # print(best_template)

    if best_score < threshold:
        return '?'

    return best_match

def inverse_threshold(image, threshold=220):
    # All values below threshold become 255 and above become 0
    binary = np.where(image < threshold, 255, 0)
    return binary.astype(np.uint8)

def ocr(image, templates, mode='path'):
    # Load templates
    if mode == 'path':
        binary_image = preprocess_image(image)
    else:
        image -= image.min()
        image = image / image.max() * 255
        binary_image = inverse_threshold(image)

    regions = extract_regions(binary_image)
    result = ''

    for (x, y, w, h) in regions:
        region = binary_image[y:y + h, x:x + w]
        char = match_template(region, templates)
        result += char

    return result


if __name__ == "__main__":
    # img = Image.open("./bra16/image_128.png").convert("L")
    # img_array = np.array(img) / 255.0
    # print(img_array.shape)
    #
    # result, conf = image2sequence(img_array)
    #
    # print(result)
    # print(conf)
    # print(isValid(result))
    templates = {
        '(': preprocess_image('templates/template(.png'),
        ')': preprocess_image('templates/template).png'),
        '[': preprocess_image('templates/template[.png'),
        ']': preprocess_image('templates/template].png'),
        '{': preprocess_image('templates/template{.png'),
        '}': preprocess_image('templates/template}.png')
    }
    output = ocr("./bra16_threepara_ratio0.2_data/image_258.png", templates)
    print(output)