import ast
import os
import pathlib
from dotenv import load_dotenv

import neptune
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDatasetSingle, ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.faster_RCNN import get_faster_rcnn_resnet
from pytorch_faster_rcnn_tutorial.transformations import ComposeDouble
from pytorch_faster_rcnn_tutorial.transformations import ComposeSingle
from pytorch_faster_rcnn_tutorial.transformations import FunctionWrapperDouble
from pytorch_faster_rcnn_tutorial.transformations import FunctionWrapperSingle
from pytorch_faster_rcnn_tutorial.transformations import apply_nms, apply_score_threshold
from pytorch_faster_rcnn_tutorial.transformations import normalize_01
from pytorch_faster_rcnn_tutorial.utils import get_filenames_of_path, collate_single, save_json
from pytorch_faster_rcnn_tutorial.visual import DatasetViewer
from pytorch_faster_rcnn_tutorial.visual import DatasetViewerSingle
from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones


# %%
# parameters
params = {'EXPERIMENT': 'FRCNN-24',  # experiment name, e.g. Head-42
          'OWNER': 'svf',  # e.g. johndoe55
          'INPUT_DIR': 'pytorch_faster_rcnn_tutorial/data/stop_line/test',  # files to predict heads/test
          'PREDICTIONS_PATH': 'predictions',  # where to save the predictions
          'MODEL_DIR': 'pytorch_faster_rcnn_tutorial/data/stop_line/epoch=0-step=1.ckpt',  # load model from checkpoint
          'DOWNLOAD': False,  # whether to download from neptune
          'DOWNLOAD_PATH': 'model',  # where to save the model if DOWNLOAD is True
          'PROJECT': 'F-RCNN',  # Project name
          }

# %%
# input files
inputs = get_filenames_of_path(pathlib.Path(params['INPUT_DIR']))
inputs.sort()

# %%
# transformations
transforms = ComposeSingle([
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    FunctionWrapperSingle(normalize_01)
])

# %%
# create dataset
dataset = ObjectDetectionDatasetSingle(inputs=inputs,
                                       transform=transforms,
                                       use_cache=False,
                                       )

# %%
# create dataloader
dataloader_prediction = DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_single)

# %%
load_dotenv()
# api_key = os.environ['NEPTUNE']  # if this throws an error, you probably didn't set your env var
api_key = os.environ.get('NEPTUNE')

# %%
#api_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjYwMmIwNy1jMDZlLTRiYTYtODQyZC0zNGRkMmRjYzg2MmUifQ=="

# %%
# import experiment from neptune
project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # get project
experiment_id = params['EXPERIMENT']  # experiment id
experiment = project.get_experiments(id=experiment_id)[0]
parameters = experiment.get_parameters()
properties = experiment.get_properties()

# %%
# rcnn transform
transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
                                     max_size=int(parameters['MAX_SIZE']),
                                     image_mean=ast.literal_eval(parameters['IMG_MEAN']),
                                     image_std=ast.literal_eval(parameters['IMG_STD']))

# %%
# view dataset
# datasetviewer = DatasetViewerSingle(dataset, rccn_transform=None)
# datasetviewer.napari()

# %%
checkpoint = torch.load('pytorch_faster_rcnn_tutorial\data\stop_line\epoch=0-step=1.ckpt', map_location=torch.device('cpu'))

# %%
# download model from neptune or load from checkpoint
if params['DOWNLOAD']:
    download_path = pathlib.Path(os.getcwd()) / params['DOWNLOAD_PATH']
    download_path.mkdir(parents=True, exist_ok=True)
    model_name = 'best_model.pt'  # that's how I called the best model
    # model_name = properties['checkpoint_name']  # logged when called log_model_neptune()
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model

    model_state_dict = torch.load(download_path / model_name, map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(params['MODEL_DIR'], map_location=torch.device('cpu'))
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

# %%
# model init

# print('backbone parameters:', type(ResNetBackbones(parameters['BACKBONE'].split('.')[-1].lower())))
# print('backbone parameters:', (ResNetBackbones('resnet34')))
# input()
model = get_faster_rcnn_resnet(num_classes=int(parameters['CLASSES']),
                               backbone_name=ResNetBackbones(parameters['BACKBONE'].split('.')[-1].lower()),  # reverse look-up enum
                               anchor_size=ast.literal_eval(parameters['ANCHOR_SIZE']),
                               aspect_ratios=ast.literal_eval(parameters['ASPECT_RATIOS']),
                               fpn=ast.literal_eval(parameters['FPN']),
                               min_size=int(parameters['MIN_SIZE']),
                               max_size=int(parameters['MAX_SIZE'])
                               )

# %%
# load weights
model.load_state_dict(model_state_dict)

# %%
# inference (cpu)
model.eval()
for sample in dataloader_prediction:
    x, x_name = sample
    with torch.no_grad():
        pred = model(x)
        pred = {key: value.numpy() for key, value in pred[0].items()}
        name = pathlib.Path(x_name[0])
        save_dir = pathlib.Path(os.getcwd()) / params['PREDICTIONS_PATH']
        save_dir.mkdir(parents=True, exist_ok=True)
        pred_list = {key: value.tolist() for key, value in pred.items()}  # numpy arrays are not serializable -> .tolist()
        save_json(pred_list, path=save_dir / name.with_suffix('.json'))

# %%
# get prediction files
predictions = get_filenames_of_path(pathlib.Path(os.getcwd()) / params['PREDICTIONS_PATH'])
predictions.sort()

# %%
# create prediction dataset
iou_threshold = 0.25
score_threshold = 0.6

transforms_prediction = ComposeDouble([
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01),
    FunctionWrapperDouble(apply_nms, input=False, target=True, iou_threshold=iou_threshold),
    FunctionWrapperDouble(apply_score_threshold, input=False, target=True, score_threshold=score_threshold)
])

dataset_prediction = ObjectDetectionDataSet(inputs=inputs,
                                            targets=predictions,
                                            transform=transforms_prediction,
                                            use_cache=False)

# %%
# mapping
color_mapping = {
    1: 'red',
}

# %%
# visualize predictions
datasetviewer_prediction = DatasetViewer(dataset_prediction, color_mapping)
datasetviewer_prediction.napari()
# add text properties gui
datasetviewer_prediction.gui_text_properties(datasetviewer_prediction.shape_layer)

# %% [markdown]
# ## Experiment with Non-maximum suppression (nms) and score-thresholding

# %%
# experiment with nms and score-thresholding
transforms_prediction = ComposeDouble([
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

dataset_prediction = ObjectDetectionDataSet(inputs=inputs,
                                            targets=predictions,
                                            transform=transforms_prediction,
                                            use_cache=False)

color_mapping = {
    1: 'red',
}

datasetviewer_prediction = DatasetViewer(dataset_prediction, color_mapping)
datasetviewer_prediction.napari()

# %%
# add score slider
# DOES CURRENTLY NOT WORK
# datasetviewer_prediction.gui_score_slider(datasetviewer_prediction.shape_layer)

# %%
# add nms slider
# DOES CURRENTLY NOT WORK
# datasetviewer_prediction.gui_nms_slider(datasetviewer_prediction.shape_layer)


