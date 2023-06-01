import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Epic Kitchens and ActionNet labels and colors.

LABELS = {'EK':{
        0 : "take (get)",
        1 : "put-down (put/place)",
        2 : "open",
        3 : "close",
        4 : "wash (clean)",
        5 : "cut",
        6 : "stir (mix)",
        7 : "pour"
}, 'AN': {
        0 : "Spread",
        1 : "Get/Put",
        2 : "Clear",
        3 : "Slice",
        4 : "Clean",
        5 : "Pour",
        6 : "Load",
        7 : "Peel",
        8 : "Open/Close",
        9 : "Set",
        10 : "Stack",
        11 : "Unload"

}}

# TODO: Better palette
COLORS = {'EK': {
        0 : "#A52A2A",
        1 : "#DAA520",
        2 : "#FF7F50",
        3 : "#7BC8F6",
        4 : "#FFFF14",
        5 : "#76FF7B",
        6 : "#13EAC9",
        7 : "olive"
}, 'AN': {
        0 : "#A52A2A",
        1 : "#DAA520",
        2 : "#FF7F50",
        3 : "#7BC8F6",
        4 : "#FFFF14",
        5 : "#76FF7B",
        6 : "#13EAC9",
        7 : "olive",
        8 : "palegreen",
        9 : "lightpink",
        10 : "darkmagenta",
        11 : "cadetblue"
}}



def show_features(feature_name, modality, dataset = "EK", split = "train", n_dim = 2, method = 'tsne', model = "I3D", annotation = None, video_level = False, num_clips = 5, title = "Features"):
    """
    Plot the features of the dataset using the specified method.
    - feature_name: name of the feature to plot(.pkl file)
    - dataset: dataset to plot [EK, AN]
    - n_dim: number of dimensions to plot
    - method: method to use for dimensionality reduction [tsne, pca]
    - model: model used to extract the features [I3D]
    """
    if n_dim != 2 and n_dim != 3:
        raise ValueError("n_dim must be 2 or 3")
    if method != 'tsne' and method != 'pca':
        raise ValueError("method must be tsne or pca")
    if dataset != "EK" and dataset != "AN":
        raise ValueError("dataset must be EK or AN")
    
    label_name = "label" if annotation == None else "verb_class"
    
    # Load the features
    data = pd.DataFrame(pd.read_pickle(feature_name)['features'])
    # data_raw is always a dictionary with only one entry (features)
    if annotation is not None:
        # in this case we have to load the annotations and merge them with the features
        annotations = pd.read_pickle(annotation)
        data= pd.merge(data, annotations, how="inner", on="uid")

    # extract from each video the corrisponding feature(in this case, the middle clip)
    
    features = [ video[num_clips // 2, :] for video in data[f'features_{modality}'] ]
    print(f"Features size: {len(features)}, shape: {features[0].shape}")
    reduced = None
    if method == "tsne":
        reduced = TSNE(n_components=n_dim, random_state=0).fit_transform(features)
    else:
        reduced = PCA(n_components=n_dim).fit_transform(features)
    data['x'] = reduced[:, 0]
    data['y'] = reduced[:, 1]
    for i in range(12): # ek has 8 classes
        filtered = data[data[label_name] == i]
        plt.scatter(filtered['x'], filtered['y'], c=COLORS['AN'][i], label=LABELS['AN'][i])
    plt.title(title)
    plt.show()

def plot_reconstructed_sep(reconstructed_features, original_features, reduction = "tsne", model = "AE", num_clips = 5):
    """
    Function to plot the reconstructed features and the original ones, using PCA or TSNE.
    - reconstructed_features: path to the reconstructed features
    - original_features: path to the original features
    - reduction: ["tsne", "pca"], reduction method to use
    - model: ["AE", "VAE"], model used to reconstruct the features
    """
    fig, ax = plt.subplots(2, figsize=(10, 10))

    data_original = pd.DataFrame(pd.read_pickle(original_features)['features'])
    annotations = pd.read_pickle("train_val/D1_train.pkl")
    data_original = pd.merge(data_original, annotations, how="inner", on="uid")
    features = data_original['features_RGB']    

    features = [f[num_clips//2] for f in features]
    for i in range(len(features)):
        if len(features[i]) != 1024:
            print(f"OPS, PROBLEMA: {len(features[i])}")
    reduced = None
    pca = None
    if reduction == "tsne":
        reduced = TSNE(n_components=2, random_state=0).fit_transform(features)
    else:
        pca = PCA().fit(features)
        reduced = pca.transform(features)
    data_original['x'] = reduced[:, 0]
    data_original['y'] = reduced[:, 1]
    for i in range(8): # ek has 8 classes
        filtered = data_original[data_original["verb_class"] == i]
        ax[0].scatter(filtered['x'], filtered['y'], c=COLORS['EK'][i], label=LABELS['EK'][i])
    ax[0].title.set_text('Original features')

    data_reconstructed = pd.DataFrame(pd.read_pickle(reconstructed_features)['features'])

    features = data_reconstructed['features_RGB']
    features = [f[num_clips//2] for f in features]
    print(f"Features size: {len(features)}")
    for i in range(len(features)):
        if len(features[i]) != 1024:
            print(f"OPS, PROBLEMA: {len(features[i])}")

    reduced = None
    pca = None
    if reduction == "tsne":
        reduced = TSNE(n_components=2, random_state=0).fit_transform(features)
    else:
        pca = PCA().fit(features)
        reduced = pca.transform(features)

    data_reconstructed['x'] = reduced[:, 0]
    data_reconstructed['y'] = reduced[:, 1]
    for i in range(8): # ek has 8 classes
        filtered = data_reconstructed[data_reconstructed["label"] == i]
        ax[1].scatter(filtered['x'], filtered['y'], c=COLORS['EK'][i], label=LABELS['EK'][i])
    ax[1].title.set_text('Reconstructed features')
    fig.suptitle(f"Feature reconstructed with {model}")
    fig.show()

def plot_reconstructed(reconstructed_features, original_features, reduction = "tsne", model = "AE", num_clips = 5):
    """
    Function to plot the reconstructed features and the original ones, using PCA or TSNE.
    - reconstructed_features: path to the reconstructed features
    - original_features: path to the original features
    - reduction: ["tsne", "pca"], reduction method to use
    - model: ["AE", "VAE"], model used to reconstruct the features
    """
    fig, ax = plt.subplots(2, figsize=(10, 10))

    data_original = pd.DataFrame(pd.read_pickle(original_features)['features'])
    annotations = pd.read_pickle("train_val/D1_train.pkl")
    data_original = pd.merge(data_original, annotations, how="inner", on="uid")
    features_original = data_original['features_RGB']    
    nof = len(features_original)

    features_original = [f[num_clips//2] for f in features_original]
    for i in range(len(features_original)):
        if len(features_original[i]) != 1024:
            print(f"OPS, PROBLEMA: {len(features_original[i])}")

    data_reconstructed = pd.DataFrame(pd.read_pickle(reconstructed_features)['features'])

    features_reconstructed = data_reconstructed['features_RGB']
    features_reconstructed = [f[num_clips//2] for f in features_reconstructed]
    print(f"Features size: {len(features_reconstructed)}")
    for i in range(len(features_reconstructed)):
        if len(features_reconstructed[i]) != 1024:
            print(f"OPS, PROBLEMA: {len(features_reconstructed[i])}")

    features = features_original + features_reconstructed

    reduced = None
    pca = None
    if reduction == "tsne":
        reduced = TSNE(n_components=2, random_state=42).fit_transform(features)
    else:
        pca = PCA().fit(features)
        reduced = pca.transform(features)
    data_original['x'] = reduced[:nof, 0]
    data_original['y'] = reduced[:nof, 1]    
    data_reconstructed['x'] = reduced[nof:, 0]
    data_reconstructed['y'] = reduced[nof:, 1]
    
    for i in range(8): # ek has 8 classes
        filtered = data_original[data_original["verb_class"] == i]
        ax[0].scatter(filtered['x'], filtered['y'], c=COLORS['EK'][i], label=LABELS['EK'][i])
    ax[0].title.set_text('Original features')

    for i in range(8): # ek has 8 classes
        filtered = data_reconstructed[data_reconstructed["label"] == i]
        ax[1].scatter(filtered['x'], filtered['y'], c=COLORS['EK'][i], label=LABELS['EK'][i])
    ax[1].title.set_text('Reconstructed features')
    fig.suptitle(f"Feature reconstructed with {model}")
    fig.show()