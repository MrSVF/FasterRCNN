# imports
import ast
import pathlib

import neptune
import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_faster_rcnn_tutorial.api_key_neptune import get_api_key
from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDatasetSingle, ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, ComposeDouble, FunctionWrapperDouble
from pytorch_faster_rcnn_tutorial.utils import get_filenames_of_path, collate_single

# parameters
params = {'EXPERIMENT': 'experiment_name',
          'INPUT_DIR': "...\test", # files to predict
          'PREDICTIONS_PATH': '...\predictions', # where to save the predictions
          'MODEL_DIR': '...\Heads', # load model from checkpoint
          'DOWNLOAD': False, # wether to download from neptune
          'DOWNLOAD_PATH': '...\Heads', # where to save the model
          'OWNER': 'your_neptune_name',
          'PROJECT': 'Heads',
          }

# input files
inputs = get_filenames_of_path(pathlib.Path(params['INPUT_DIR']))
inputs.sort()

# transformations
transforms = ComposeSingle([
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    FunctionWrapperSingle(normalize_01)
])

# create dataset and dataloader
dataset = ObjectDetectionDatasetSingle(inputs=inputs,
                                       transform=transforms,
                                       use_cache=False,
                                       )

dataloader_prediction = DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_single)

# import experiment from neptune
api_key = get_api_key()  # get the personal api key
project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # get project
experiment_id = params['EXPERIMENT']  # experiment id
experiment = project.get_experiments(id=experiment_id)[0]
parameters = experiment.get_parameters()
properties = experiment.get_properties()

# view dataset
from visual import DatasetViewerSingle
from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
                                     max_size=int(parameters['MAX_SIZE']),
                                     image_mean=ast.literal_eval(parameters['IMG_MEAN']),
                                     image_std=ast.literal_eval(parameters['IMG_STD']))


datasetviewer = DatasetViewerSingle(dataset, rccn_transform=None)
datasetviewer.napari()

# download model from neptune or load from checkpoint
if params['DOWNLOAD']:
    download_path = pathlib.Path(params['DOWNLOAD_PATH'])
    model_name = properties['checkpoint_name'] # logged when called log_model_neptune()
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model

    model_state_dict = torch.load(download_path / model_name)
else:
    checkpoint = torch.load(params['MODEL_DIR'])
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

# model init
from faster_RCNN import get_fasterRCNN_resnet
model = get_fasterRCNN_resnet(num_classes=int(parameters['CLASSES']),
                              backbone_name=parameters['BACKBONE'],
                              anchor_size=ast.literal_eval(parameters['ANCHOR_SIZE']),
                              aspect_ratios=ast.literal_eval(parameters['ASPECT_RATIOS']),
                              fpn=ast.literal_eval(parameters['FPN']),
                              min_size=int(parameters['MIN_SIZE']),
                              max_size=int(parameters['MAX_SIZE'])
                              )

# load weights
model.load_state_dict(model_state_dict)