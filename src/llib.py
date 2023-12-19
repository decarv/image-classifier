import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import os
import csv
from typing import Callable
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from PIL import Image, ImageEnhance
from itertools import product
import pickle


#####################
#  GLOBAL VARIABLES #
#####################

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
AUGMENTED_DATASET_DIR = os.path.join(PROJECT_DIR, "augmented_dataset")
NORMALIZED_DATASET_DIR = os.path.join(PROJECT_DIR, "normalized_dataset")
SEGMENTED_DATASET_DIR = os.path.join(PROJECT_DIR, "segmented_dataset")

METADATA_FILEPATH = os.path.join(DATASET_DIR, "metadados.csv")
SUMMARY_FILEPATH = os.path.join(DATASET_DIR, "sumario.csv")

CLASSIFIER_FILEPATH = os.path.join(PROJECT_DIR, "classifier.pkl")

MAX_IMAGES = 100
IMAGE_FILENAME = "{}_{}.png"

BACKGROUNDS = ["Branco"]

#####################
#        LIB        #
#####################


class Item:
    def __init__(self, name, ui_name, index):
        self.name = name
        self.ui_name = ui_name
        self.index = index
        self.path = os.path.join(DATASET_DIR, name)

    def __repr__(self):
        return self.ui_name


class Collection:
    def __init__(self):
        self.items = {}
        self.count = 0

    def add_item(self, name, ui_name):
        self.count += 1
        self.items[name] = Item(name, ui_name, self.count)

    def keys(self):
        return list(self.items.keys())

    def get(self, cls_name):
        return self.items.get(cls_name)

    def ui_names(self):
        return [item.ui_name for item in self.items.values()]


class Classifier:
    def __init__(self):
        self.segmentation_params = (9, 75, 50, 3, 2)
        with open(CLASSIFIER_FILEPATH, 'rb') as file:
            self.classifier = pickle.load(file)

    def detect(self, image, **kwargs):
        image = rgb2gray(image)
        transformed_images = [log_transform(image), exp_transform(image), mean_filter(image), gray_gradient(image)]
        normalized_images = [histogram_equalization(im) for im in transformed_images]
        segmented_images = [create_segmentation_mask(im, *self.segmentation_params) for im in normalized_images]
        features = []
        for im in segmented_images:
            if im is not None:
                feat = extract_features(im)
                if feat is not None:
                    features.append(feat)
        features = pd.DataFrame(features)

        if not features.empty:
            predictions = self.classifier.predict(features)
            best_score_index = np.argmax(predictions)
            best_image = segmented_images[best_score_index]
            class_name = predictions[best_score_index]
            x, y, w, h = get_bounding_box(best_image)
            return [class_name], [(x, y, w, h)]
        else:
            return image, None


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


def get_id(image):
    """
    Retorna o id do objeto a partir da imagem.
    """
    return str(abs(hash(image.tobytes())))


def read_metadata():
    metadata = {}
    with open(METADATA_FILEPATH, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                metadata[row[0]] = row[1:]
    return metadata


def update_metadata(filename, obj_id, name, bg_color):
    """
    Carrega o arquivo de metadados e adiciona uma nova linha com o nome do arquivo e a classe.

    colunas metadados.csv:
        filename, obj_id, name, bg_color
    """
    with open(METADATA_FILEPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, obj_id, name, bg_color])


def delete_metadata(filename):
    """
    Carrega o arquivo de metadados e remove a linha com o nome do arquivo.

    colunas metadados.csv:
        filename, obj_id, name, bg_color
    """
    with open(METADATA_FILEPATH, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader if row and row[0] != filename]
    with open(METADATA_FILEPATH, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)


def read_dataset(parent_dir: str = DATASET_DIR) -> dict[str, list[np.ndarray]]:
    images = {}
    for d in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, d)):
            images[d] = []
            for imfp in os.listdir(os.path.join(parent_dir, d)):
                try:
                    img_path = os.path.join(parent_dir, d, imfp)
                    arr = np.load(img_path).reshape(1290, -1)
                except ValueError:
                    arr = cv2.imread(img_path)
                images[d].append(arr)
    return images


def write_dataset(dataset: dict[str, list[np.ndarray]], parent_dir: str):
    for cls in dataset.keys():
        for arr in dataset[cls]:
            directory = os.path.join(parent_dir, cls)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            fp = os.path.join(directory, f"{cls}_{get_id(arr)}")
            np.save(fp, arr)


def iaa_transform(transform_function: Callable):
    def lambda_transformation(images, random_state, parents, hooks):
        return [transform_function(image) for image in images]

    return iaa.Lambda(func_images=lambda_transformation)


def transformations2seq(transformations):
    return iaa.Sequential([iaa_transform(transform) for transform in transformations])


def apply_transformations_to_dataset(dataset, transformations):
    transformed_dataset = {k: [] for k in dataset.keys()}
    for transformation in transformations:
        pipeline = transformations2seq([transformation])
        new_dataset = apply_pipeline_to_dataset(dataset, pipeline)
        for k in new_dataset:
            transformed_dataset[k].extend(new_dataset[k])
    return transformed_dataset


def apply_pipeline_to_dataset(dataset, pipeline: iaa.Sequential, n: int = 1):
    new_dataset = {k: [] for k in dataset.keys()}
    for k in dataset:
        for _ in range(n):
            images = dataset[k]
            for i, im in enumerate(images):
                if im.ndim == 2:  # If the image is grayscale
                    images[i] = im[:, :, np.newaxis]  # Add a third dimension
            transformed_images = pipeline(images=images)
            try:
                transformed_images = [im.reshape(1290, -1) for im in transformed_images]
            except ValueError:
                pass
            new_dataset[k].extend(transformed_images)
    return new_dataset


def rgb2gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gray_gradient(image):
    h, w = image.shape[:2]
    gradient = np.linspace(0, 255, w, dtype=np.uint8)
    gradient = np.tile(gradient, (h, 1))
    return cv2.addWeighted(image, 0.5, gradient, 0.5, 0)


def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image)
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed


def exp_transform(image):
    c = 255 / (np.exp(1) - 1)
    exp_transformed = c * (np.exp(image / 255) - 1)
    exp_transformed = np.array(exp_transformed, dtype=np.uint8)
    return exp_transformed


def mean_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(image, -1, kernel)
    return cv2.blur(img, (5, 5))


def plot_mnist(dataset, num_columns=10):
    max_images = 100
    images = []
    for class_images in dataset.values():
        for im in class_images:
            images.append(im)
            if len(images) == max_images:
                break

    num_samples = len(images)
    num_rows = num_samples // num_columns + (num_samples % num_columns > 0)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows))
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.imshow(images[i], aspect='auto', cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
            ax.set_frame_on(False)

    plt.tight_layout()


def get_bounding_box(image):
    _, thresh = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x-2, y-2, w+4, h+4)

def draw_bounding_box(image):
    _, thresh = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    bounding_box_image = cv2.rectangle(color_image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 0), 2)

    extracted_region = image[y - 2:y + h + 2, x - 2:x + w + 2]

    return bounding_box_image, extracted_region


def histogram_equalization(image):
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        raise ValueError("Image must be grayscale")


def normalize_dataset(dataset):
    normalized_dataset = {}
    for k, images in dataset.items():
        try:
            normalized_dataset[k] = [histogram_equalization(img) for img in images]
        except ValueError:
            tmp = []
            for img in images:
                if img.ndim == 3:
                    assert img.shape[2] == 1
                    img = img.reshape(img.shape[0], -1)
                    tmp.append(histogram_equalization(img))
            normalized_dataset[k] = tmp

    return normalized_dataset


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.iloc[:, :-1].quantile(0.25)
    Q3 = df.iloc[:, :-1].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df.iloc[:, :-1] < (Q1 - 1.5 * IQR)) | (df.iloc[:, :-1] > (Q3 + 1.5 * IQR))
    return df[~outlier_mask.any(axis=1)]


def create_segmentation_mask(image, filter_diameter, sigma_color, sigma_space, kernel_size, iterations):
    if len(image.shape) == 3 and image.shape[2] == 3:
        raise ValueError("Expected a grayscale image")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(1290, -1)

    blur = cv2.bilateralFilter(image, filter_diameter, sigma_color, sigma_space)

    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary_mask)
    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask


def create_ground_truth_grid(dataset, filter_diameters, sigma_colors, sigma_spaces, kernel_sizes, iterations):
    ground_truth_grid = []
    for filter_diameter, sigma_color, sigma_space, kernel_size, iteration in product(
            filter_diameters, sigma_colors, sigma_spaces, kernel_sizes, iterations
    ):
        ground_truth_dataset = []
        for data in dataset:
            ground_truth_dataset.append({
                'image': create_segmentation_mask(data['image'], filter_diameter, sigma_color, sigma_space,
                                                  kernel_size, iteration),
                'class': data['class'],
            })
        ground_truth_grid.append({
            'dataset': ground_truth_dataset,
            'parameters': (filter_diameter, sigma_color, sigma_space, kernel_size, iteration),
        })

    return ground_truth_grid


def extract_features(binary_image):
    assert isinstance(binary_image, np.ndarray)

    _, thresh = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)

    # Área
    area = cv2.contourArea(largest_contour)

    # Perímetro
    perimeter = cv2.arcLength(largest_contour, True)

    # Bounding Rect
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area

    # Convex Hull
    try:
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
    except ZeroDivisionError:
        solidity = 0

    # Diâmetro
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # Orientação e Excentricidade
    moments = cv2.moments(largest_contour)
    if moments['mu20'] + moments['mu02'] != 0:
        eccentricity = np.sqrt(2 * (moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) / (
                    moments['mu20'] + moments['mu02'])
        orientation = 0.5 * np.arctan((2 * moments['mu11']) / (moments['mu20'] - moments['mu02']))
    else:
        eccentricity = 0
        orientation = 0

    # Compactação
    if area != 0:
        compactness = (perimeter ** 2) / area
    else:
        compactness = 0

    features = {
        "Area": area,
        "Perimeter": perimeter,
        "Aspect Ratio": aspect_ratio,
        "Extent": extent,
        "Solidity": solidity,
        "Equivalent Diameter": equivalent_diameter,
        "Orientation": orientation,
        "Eccentricity": eccentricity,
        "Compactness": compactness
    }

    return features


def eval_classifiers(dfs: list[dict[str, list | pd.DataFrame]]) -> list[dict]:
    scores = []
    for data in dfs:
        params = data['params']
        df = data['df']
        scores_dict = {'params': params, 'best': {}, 'classifiers': {}, 'scores': []}

        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])

        X = df.drop('class', axis=1)
        y = df['class']

        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        X_new = selector.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

        classifiers = {
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(),
            'RandomForest': RandomForestClassifier(),
            'DecisionTree': DecisionTreeClassifier(),
            'LogisticRegression': LogisticRegression()
        }

        best_name: str = ""
        best_accuracy: float = 0.0
        for name, clf in classifiers.items():
            scores_dict['classifiers'][name] = {}
            clf.fit(X_train, y_train)
            try:
                y_pred = clf.predict(X_test)
                scores_dict['classifiers'][name]['acc'] = accuracy_score(y_test, y_pred)
                scores_dict['classifiers'][name]['report'] = classification_report(y_test, y_pred)
            except ValueError:
                scores_dict['classifiers'][name]['acc'] = 0.0
                scores_dict['classifiers'][name]['report'] = 0.0

            if best_accuracy < scores_dict['classifiers'][name]['acc']:
                best_accuracy = scores_dict['classifiers'][name]['acc']
                best_name = name

        scores_dict['scores'] = selector.scores_
        scores_dict['best'] = {'name': best_name, 'acc': best_accuracy, 'params': params}
        scores.append(scores_dict)
    return scores

def create_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    features = []
    for index, row in df.iterrows():
        f = extract_features(row['image'])
        if f is None:
            continue
        features.append(f)
        features[-1]['class'] = row['class']
    return pd.DataFrame(features)


def store_best_random_forest(dfs: list[dict[str, list | pd.DataFrame]], save_path) -> list[dict]:
    scores = []
    best_overall_classifier = None
    highest_accuracy = 0
    best_overall_params = None
    best_overall_importances = None

    for data in dfs:
        params = data['params']
        df = data['df']
        scores_dict = {'params': params, 'classifiers': {}}

        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])

        X = df.drop('class', axis=1)
        y = df['class']

        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(X, y)

        sfm = SelectFromModel(forest, threshold=0.01)
        X_transformed = sfm.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3)

        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        feature_importances = forest.feature_importances_[sfm.get_support()]

        scores_dict['classifiers']['RandomForest'] = {
            'acc': accuracy,
            'report': report,
            'feature_importances': feature_importances
        }

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_overall_classifier = forest
            best_overall_params = params
            best_overall_importances = feature_importances

        scores.append(scores_dict)

    if best_overall_classifier:
        with open(save_path, 'wb') as file:
            pickle.dump(best_overall_classifier, file)

    best_classifier_info = {
        'accuracy': highest_accuracy,
        'params': best_overall_params,
        'feature_importances': list(best_overall_importances)
    }

    return scores, best_classifier_info
