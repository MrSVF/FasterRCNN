# imports
import os
from dotenv import load_dotenv
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import albumentations as albu
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch.utils.data import DataLoader

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.faster_RCNN import (
    FasterRCNNLightning,
    get_faster_rcnn_resnet,
)
from pytorch_faster_rcnn_tutorial.transformations import (
    AlbumentationWrapper,
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_double,
    get_filenames_of_path,
    log_mapping_neptune,
    log_model_neptune,
    log_packages_neptune,
)


# hyper-parameters
@dataclass
class Params:
    BATCH_SIZE: int = 1#2
    OWNER: str = "svf"  # set your name here, e.g. johndoe22, было: "johschmidt42"
    SAVE_DIR: Optional[
        str
    ] = None  # checkpoints will be saved to cwd (current working directory)
    LOG_MODEL: bool = False  # whether to log the model to neptune after training
    GPU: Optional[int] = 1 #None  # set to None for cpu training
    LR: float = 0.001
    PRECISION: int = 32
    CLASSES: int = 3
    SEED: int = 42
    PROJECT: str = "F-RCNN"
    EXPERIMENT: str = "SpeedBump"
    MAXEPOCHS: int = 500
    PATIENCE: int = 50
    BACKBONE: ResNetBackbones = ResNetBackbones.RESNET34
    FPN: bool = False#True
    ANCHOR_SIZE: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),) #((32, 64, 128, 256, 512),) #(16,), (32,), (64,), (128,), (256,), (512,)
    ASPECT_RATIOS: Tuple[Tuple[float, ...]] = ((0.5, 1.0, 2.0),)
    MIN_SIZE: int = 1440 #1024
    MAX_SIZE: int = 1441 #1025
    IMG_MEAN: List = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMG_STD: List = field(default_factory=lambda: [0.229, 0.224, 0.225])
    IOU_THRESHOLD: float = 0.5


# root directory
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


def main():
    params = Params()

    load_dotenv()
    # api key
    # api_key = os.environ[
    #     "NEPTUNE"
    # ]  # if this throws an error, you didn't set your env var
    api_key = os.environ.get('NEPTUNE')

    # save directory
    save_dir = os.getcwd() if not params.SAVE_DIR else params.SAVE_DIR

    # root directory
    root = ROOT_PATH / "pytorch_faster_rcnn_tutorial" / "data" / "speed_bump"
    # root = ROOT_PATH / "pytorch_faster_rcnn_tutorial" / "data" / "heads" 

    # input and target files
    inputs = get_filenames_of_path(root / "input2")
    targets = get_filenames_of_path(root / "target2")

    inputs.sort()
    targets.sort()

    # mapping
    mapping = {
        "speedbump": 1,
        "bumpsign": 2,
    }

    # training transformations and augmentations
    transforms_training = ComposeDouble(
        [
            Clip(),
            AlbumentationWrapper(albumentation=albu.HorizontalFlip(p=0.5)),
            AlbumentationWrapper(
                albumentation=albu.RandomScale(p=0.5, scale_limit=0.5)
            ),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # validation transformations
    transforms_validation = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # test transformations
    transforms_test = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01),
        ]
    )

    # random seed
    seed_everything(params.SEED)


    train_ind = int(len(inputs)*0.6)
    valid_ind = train_ind + int(len(inputs)*0.2)
    # train_ind = 60
    # valid_ind = 70
    # training validation test split
    inputs_train, inputs_valid, inputs_test = inputs[:train_ind], inputs[train_ind:valid_ind], inputs[valid_ind:] #inputs[:12], inputs[12:16], inputs[16:]
    targets_train, targets_valid, targets_test = (
        targets[:train_ind],
        targets[train_ind:valid_ind],
        targets[valid_ind:],
    )
    # print(train_ind, valid_ind)
    # input()
    # print('targets_train:', targets_train[0])
    # input()

    # dataset training
    dataset_train = ObjectDetectionDataSet(
        inputs=inputs_train,
        targets=targets_train,
        transform=transforms_training,
        use_cache=False, #True
        convert_to_format=None,
        mapping=mapping,
    )

    # dataset validation
    dataset_valid = ObjectDetectionDataSet(
        inputs=inputs_valid,
        targets=targets_valid,
        transform=transforms_validation,
        use_cache=False, #True
        convert_to_format=None,
        mapping=mapping,
    )

    # dataset test
    dataset_test = ObjectDetectionDataSet(
        inputs=inputs_test,
        targets=targets_test,
        transform=transforms_test,
        use_cache=False, #True
        convert_to_format=None,
        mapping=mapping,
    )

    # dataloader training
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_double,
    )

    # dataloader validation
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # dataloader test
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # neptune logger
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=f"{params.OWNER}/{params.PROJECT}",  # use your neptune name here
        experiment_name=params.PROJECT,
        params=params.__dict__,
    )

    assert neptune_logger.name  # http GET request to check if the project exists

    # model init
    model = get_faster_rcnn_resnet(
        num_classes=params.CLASSES,
        backbone_name=params.BACKBONE,
        anchor_size=params.ANCHOR_SIZE,
        aspect_ratios=params.ASPECT_RATIOS,
        fpn=params.FPN,
        min_size=params.MIN_SIZE,
        max_size=params.MAX_SIZE,
    )
    model
    # lightning init
    task = FasterRCNNLightning(
        model=model, lr=params.LR, iou_threshold=params.IOU_THRESHOLD
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor="Validation_mAP", mode="max")
    learningrate_callback = LearningRateMonitor(
        logging_interval="step", log_momentum=False
    )
    early_stopping_callback = EarlyStopping(
        monitor="Validation_mAP", patience=params.PATIENCE, mode="max"
    )

    # trainer init
    trainer = Trainer(
        gpus=params.GPU,
        precision=params.PRECISION,  # try 16 with enable_pl_optimizer=False
        callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        default_root_dir=save_dir,  # where checkpoints are saved to
        logger=neptune_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=params.MAXEPOCHS,
    )

    # start training
    trainer.fit(
        model=task, train_dataloader=dataloader_train, val_dataloaders=dataloader_valid
    )

    # start testing
    trainer.test(ckpt_path="best", dataloaders=dataloader_test)

    # log packages
    log_packages_neptune(neptune_logger=neptune_logger)

    # log mapping as table
    log_mapping_neptune(mapping=mapping, neptune_logger=neptune_logger)

    # log model
    if params.LOG_MODEL:
        checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
        log_model_neptune(
            checkpoint_path=checkpoint_path,
            save_directory=pathlib.Path.home(),
            name="best_model.pt",
            neptune_logger=neptune_logger,
        )

    # stop logger
    neptune_logger.experiment.stop()
    print("Finished")


if __name__ == "__main__":
    main()
