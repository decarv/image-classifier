import os
from typing import Optional

import toga
from plyer import camera

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

METADATA_PATH = os.path.join(DATASET_DIR, "metadados.csv")

MAX_IMAGES = 100
IMAGE_NAME = "{}_{}.jpg"

CLASSES = {
    "fork": {
        "name": "Garfo",
        "color": "red",
        "index": 1,
        "path": os.path.join(DATASET_DIR, "fork"),
    },
    "pen": {
        "name": "Caneta",
        "color": "blue",
        "index": 2,
        "path": os.path.join(DATASET_DIR, "pen"),
    },
    "bottle": {
        "name": "Garrafa",
        "color": "green",
        "index": 3,
        "path": os.path.join(DATASET_DIR, "bottle"),
    },
    "glass": {
        "name": "Copo",
        "color": "yellow",
        "index": 4,
        "path": os.path.join(DATASET_DIR, "glass"),
    },
    "ring": {
        "name": "Anel",
        "color": "purple",
        "index": 5,
        "path": os.path.join(DATASET_DIR, "ring"),
    }
}


def take_pic(widget, app, cls):
    filename = get_filename(cls)
    if filename is None:
        app.main_window.error_dialog("Error", "Diretório está cheio.")
        return None
    camera.take_picture(filename=filename, on_complete=camera_callback)


def camera_callback(filepath):
    if os.path.exists(filepath):
        print("Foto salva!")
    else:
        print("ERRO: Foto não foi tirada!")


def get_filename(cls):
    saved_files = set(os.listdir(cls["path"]))
    for i in range(MAX_IMAGES):
        filename = IMAGE_NAME.format(cls["name"], i)
        if filename not in saved_files:
            return filename
    return None


def build(app):
    box = toga.Box(style=toga.style.Pack(direction=toga.style.pack.COLUMN, padding=10))

    filename_input = toga.TextInput(placeholder="Nome do arquivo")
    class_select = toga.Selection(items=[cls["name"] for cls in CLASSES.values()])
    # image_view = toga.ImageView(id="image_view")  # Placeholder for the captured image

    def on_take_pic(widget):
        cls: Optional[str] = None
        for n in CLASSES:
            if CLASSES[n]["name"] == class_select.value:
                cls = CLASSES[n]
                break

        take_pic(widget, app, cls)

    take_pic_button = toga.Button("Tirar Foto", on_press=on_take_pic)
    cancel_button = toga.Button("Fechar", on_press=lambda widget: app.main_window.close())

    box.add(class_select)
    box.add(take_pic_button)
    # box.add(image_view)

    return box


if __name__ == "__main__":
    app = toga.App("Classificador de Imagens", "some.id", startup=build)
    app.main_loop()
