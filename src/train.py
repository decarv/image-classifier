
from llib import *

import cv2
import numpy as np
from PIL import Image
import pandas as pd
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

original_dataset = read_dataset(DATASET_DIR)
original_gray_dataset = apply_transformations_to_dataset(original_dataset, [rgb2gray])
augmented_dataset = apply_transformations_to_dataset(original_gray_dataset, [exp_transform, log_transform, mean_filter, gray_gradient])

for k in original_gray_dataset.keys():
    augmented_dataset[k].extend(original_gray_dataset[k])

corrected_dataset = apply_transformations_to_dataset(augmented_dataset, [log_transform])
augmented_dataset = corrected_dataset

normalized_augmented_dataset = normalize_dataset(augmented_dataset)
dataset = []
for cls in normalized_augmented_dataset:
    for im in normalized_augmented_dataset[cls]:
        inst = {
            'class': cls,
            'image': im,
        }
        dataset.append(inst)
normalized_augmented_dataset = dataset

filter_diameters = [9]
sigma_colors = [75]
sigma_spaces = [50]
kernel_sizes = [3]
iterations = [2]

ground_truth_grid = create_ground_truth_grid(normalized_augmented_dataset, filter_diameters, sigma_colors, sigma_spaces, kernel_sizes, iterations)

segmented_datasets = []
for dataset in ground_truth_grid:
    segmented_datasets.append({'df': pd.DataFrame(dataset['dataset']), 'params': dataset['parameters']})

features_dfs = []
for dataset in segmented_datasets:
    features_dfs.append({
        'df': create_feature_dataframe(dataset['df']),
        'params': dataset['params']
    })


for i, dataset in enumerate(features_dfs):
    features_dfs[i]['df'] = clean_dataframe(dataset['df'])

_, _ = store_best_random_forest(features_dfs, CLASSIFIER_FILEPATH)
