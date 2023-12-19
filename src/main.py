import os
import io
import csv
import sys
from typing import Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from PIL import Image, ImageEnhance

import photos
import pyto_ui as ui
from UIKit import UIPickerView
from Foundation import NSObject, NSArray
from rubicon.objc import objc_method, SEL
from llib import *


#####################
#        MAIN       #
#####################

collection = Collection()
collection.add_item("chave_de_fenda", "A")
collection.add_item("alicate", "B")
collection.add_item("chave_inglesa", "C")
collection.add_item("chave_hexagonal", "D")
collection.add_item("martelo", "E")
classifier = Classifier()

CLASSES_MAP = {
    0: "Alicate",
    1: "Chave de Fenda",
    2: "Chave Hexagonal",
    3: "Chave Inglesa",
    4: "Martelo",
}

def run_classifier():
    classifier = Classifier()
    device = 0
    try:
        device = int(sys.argv[1])  # 0 for back camera
    except IndexError:
        pass
    cap = cv2.VideoCapture(device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        if frame is None:
            continue
        frame = cv2.autorotate(frame, device)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        class_names, objects = classifier.detect(frame)
        if objects is not None:
            for class_name, (x, y, w, h) in zip(class_names, objects):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if x > 0 and y > 0:
                    cv2.putText(frame, CLASSES_MAP[class_name], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)


def _take_pic(sender):
    class_name = collection.keys()[sc_classes.selected_segment]
    cls = collection.get(class_name)
    if cls is None:
        raise ValueError("Invalid class name")

    image = photos.take_photo()

    obj_id = get_id(image)
    filename = os.path.join(cls.path, IMAGE_FILENAME.format(cls.name, obj_id))
    print(filename)
    if filename is None:
        raise ValueError("Invalid filename")

    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()
    with open(filename, 'wb') as f:
        f.write(image_byte_array)

    width, height = image.size

    update_metadata(
        filename,
        obj_id,
        cls.ui_name,
        bg_color=BACKGROUNDS[0],
    )
    update_summary(cls.ui_name, len(image_byte_array), width, height)



def update_summary(cls_name, image_size, width, height):
    """
    """
    summary_data = {}
    try:
        with open(SUMMARY_FILEPATH, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    summary_data[row[0]] = row[1]
    except FileNotFoundError:
        pass

    summary_data["Numero de Classes"] = int(summary_data.get("Numero de Classes", 0))
    summary_data["Numero de Imagens"] = int(summary_data.get("Numero de Imagens", 0))
    summary_data["Tamanho da Base"] = float(summary_data.get("Tamanho da Base", 0.0))
    summary_data["Resolucao das Imagens"] = f"({width}, {height})"

    if summary_data.get(cls_name, None) is None:
        summary_data[cls_name] = 0
        summary_data["Numero de Classes"] = int(summary_data["Numero de Classes"]) + 1
    print(summary_data['Tamanho da Base'])
    summary_data[cls_name] = int(summary_data[cls_name]) + 1
    summary_data["Numero de Imagens"] = int(summary_data['Numero de Imagens']) + 1
    summary_data["Tamanho da Base"] = f"{float(summary_data['Tamanho da Base']) + image_size / 1e6:.2f}"

    with open(SUMMARY_FILEPATH, 'w', newline='') as file:
        writer = csv.writer(file)
        for lin, val in summary_data.items():
            writer.writerow([lin, val])


def did_change(item):
    view.title = str(item)


if __name__ == "__main__":
    view = ui.View()
    view.background_color = ui.COLOR_SYSTEM_BACKGROUND

    take_pic_button = ui.Button(title="Tirar foto")
    take_pic_button.size = (200, 100)
    take_pic_button.center = (view.width / 2 + 25, view.height / 2)

    sc_classes = ui.SegmentedControl(collection.ui_names())
    sc_classes.size = (400, 50)
    sc_classes.center = (view.width / 2 + 25, view.height / 2 + 200)

    class_button = ui.Button(title="Classificar")
    class_button.size = (200, 100)
    class_button.center = (view.width / 2 + 25, view.height / 2 - 200)

    # list_picker = ListPicker()
    # # list_picker.items = collection.ui_names()
    # list_picker.did_change = did_change
    # list_picker.center = (view.width / 2 + 25, view.height / 2 + 300)

    view.add_subview(take_pic_button)
    view.add_subview(sc_classes)
    view.add_subview(class_button)

    take_pic_button.action = _take_pic
    class_button.action = run_classifier

    ui.show_view(view, ui.PRESENTATION_MODE_FULLSCREEN)
