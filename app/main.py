from flask import Flask, request, jsonify
import base64
import cv2
import torch
from torchvision import transforms
from app.tiny_face_detector.tiny_fd import TinyFacesDetector
import numpy as np
import os

tiny_detector = TinyFacesDetector(model_root='./app/tiny_face_detector/',
                                  prob_thresh=0.5, gpu_idx=0)

models_names = [
    'Mobilenetv2_on_gans_data_pretrained',
    'Shufflenetv2_on_gans_data_pretrained',
]
classes = ['with_mask', 'without_mask']
models = {
    model_name: torch.load(f'./app/models/{model_name}.pt',
                           map_location=torch.device('cpu'))
    for model_name in models_names
}


def crop_face(image_bytes):
    img = data_uri_to_cv2_img(image_bytes)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    resized_img = cv2.resize(img, (112, 112))
    faces = tiny_detector.detect(resized_img)

    for face in faces:
        for i in range(len(face)):
            face[i] = face[i] * 2
        xmin, ymin, xmax, ymax = increase_bounding_box(face, (224, 224))
        cropped_img = cv2.resize(
            cv2.cvtColor(img[ymin:ymax, xmin:xmax],
                         cv2.COLOR_RGB2BGR), (224, 224))
        return cropped_img, [xmin*480/224, ymin*480/224,
                             xmax*480/224, ymax*480/224]


def transform_image(img):
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return my_transforms(img).unsqueeze(0)


def increase_bounding_box(bndbox, target_img_size, ratio=0.15):
    xmin = bndbox[0]-int((bndbox[2]-bndbox[0])*ratio)
    xmax = bndbox[2]+int((bndbox[2]-bndbox[0])*ratio)
    ymin = bndbox[1]-int((bndbox[3]-bndbox[1])*ratio)
    ymax = bndbox[3]+int((bndbox[3]-bndbox[1])*ratio)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > target_img_size[0]:
        xmax = target_img_size[0] - 1
    if ymax > target_img_size[1]:
        ymax = target_img_size[1] - 1

    return xmin, ymin, xmax, ymax


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


app = Flask(__name__)
app.config['DEBUG'] = os.environ.get('DEBUG', 'True') == 'True'

@app.route("/", methods=['POST', 'GET'])
def classify():
    if request.json is None:
        return '', 400
    image = request.json.get('image', None)
    model_name = request.json.get('model_name', None)
    if image is None:
        return '', 400  # missing arguments
    else:
        cropped_face, bounds = crop_face(image)
        cropped_face = transform_image(cropped_face)
        prediction = models.get(model_name,
                                'Mobilenetv2_on_gans_data_pretrained'
                                ).forward(cropped_face)
        _, y_hat = prediction.max(1)
        return jsonify({
                        'prediction': classes[y_hat.item()],
                        'bounds': [int(bound) for bound in bounds]
                        })
