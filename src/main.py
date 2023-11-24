import os
import io
import photos
import datetime
import csv
import pyto_ui as ui

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

METADATA_FILEPATH = os.path.join(DATASET_DIR, "metadados.csv")
SUMMARY_FILEPATH = os.path.join(DATASET_DIR, "sumario.csv")

MAX_IMAGES = 100
IMAGE_FILENAME = "{}_{}.png"


class Class:
    def __init__(self, name, ui_name, color, index):
        self.name = name
        self.ui_name = ui_name
        self.color = color
        self.index = index
        self.path = os.path.join(DATASET_DIR, name)
        
    def __repr__(self):
        return self.ui_name


CLASSES = {
    "fork": Class("fork", "Garfo", "red", 1),
    "knife": Class("knife", "Faca", "blue", 2),
    "spoon": Class("spoon", "Colher", "green", 3),
    "cup": Class("cup", "Copo", "yellow", 4),
    "tool": Class("tool", "Ferramenta", "purple", 5),
}

BACKGROUNDS = ["Branco", "Preto", "Colorido"]

def _take_pic(sender):
    class_name = sc_classes.selected_segment()
    cls = CLASSES.get(class_name)
    if cls is None:
        raise ValueError("Invalid class name")
    obj_id = hash(str(datetime.datetime.now()).encode())
    filename = IMAGE_FILENAME.format(cls.name, obj_id)
    if filename is None:
        raise ValueError("Invalid filename")

    image = photos.take_photo()
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
        bg_color=sc_bg_color.selected_segment(),
    )
    update_summary(cls.ui_name, len(image_byte_array), width, height)

def update_metadata(filename, obj_id, name, bg_color):
    """
    Carrega o arquivo de metadados e adiciona uma nova linha com o nome do arquivo e a classe.

    colunas metadados.csv:
        filename, obj_id, name, bg_color
    """
    with open(METADATA_FILEPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, obj_id, name, bg_color])


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

    summary_data["Número de Classes"] = int(summary_data.get("Número de Classes", 0))
    summary_data["Número de Imagens"] = int(summary_data.get("Número de Imagens", 0))
    summary_data["Tamanho da Base"] = float(summary_data.get("Tamanho da Base (MB)", 0.0))
    summary_data["Resolução das Imagens"] = f"({width}, {height})"

    if summary_data.get(cls_name, None) is None:
        summary_data[cls_name] = 0
        summary_data["Número de Classes"] = summary_data["Número de Classes"] + 1

    summary_data[cls_name] += 1
    summary_data["Número de Imagens"] += 1
    summary_data["Tamanho da Base"] = f"{summary_data['Tamanho da Base'] + image_size / 1e6:.2f}"

    with open(SUMMARY_FILEPATH, 'w', newline='') as file:
        writer = csv.writer(file)
        for lin, val in summary_data.items():
            writer.writerow([lin, val])



view = ui.View()
view.background_color = ui.COLOR_SYSTEM_BACKGROUND

take_pic_button = ui.Button(title="Tirar foto")
take_pic_button.size = (200, 100)
take_pic_button.center = (view.width / 2 + 50, view.height / 2)

sc_classes = ui.SegmentedControl([str(cls) for cls in CLASSES])
sc_classes.size = (300, 50)
sc_classes.center = (view.width / 2 + 50, view.height / 2 + 200)

sc_bg_color = ui.SegmentedControl(BACKGROUNDS)
sc_classes.size = (300, 50)
sc_classes.center = (view.width / 2 + 50, view.height / 2 + 400)
#
# switch_daytime = ui.Switch(["Dia", "Noite"])
# switch_daytime.size = (100, 50)
# switch_daytime.center = (view.width / 2 + 50, view.height / 2 + 600)
#
# switch_indoor = ui.Switch(["Indoor", "Outdoor"])
# switch_indoor.size = (100, 50)
# switch_indoor.center = (view.width / 2 + 50, view.height / 2 + 800)

# view.add_subview(switch_daytime)
view.add_subview(take_pic_button)
view.add_subview(sc_classes)
view.add_subview(sc_bg_color)

take_pic_button.action = _take_pic

ui.show_view(view, ui.PRESENTATION_MODE_FULLSCREEN)

