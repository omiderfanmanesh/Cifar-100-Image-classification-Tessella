from yacs.config import CfgNode as CN
from solver.optimizer_type import OptimizerType
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# BASIC
# -----------------------------------------------------------------------------
_C.BASIC = CN()
_C.BASIC.SEED = 2021
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.NUM_GPU = 1
_C.MODEL.NAME = 'CIFAR100'
_C.MODEL.PRE_TRAINED = True
_C.MODEL.OPTIMIZER = OptimizerType.SGD
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image.jpg during training
_C.INPUT.SIZE_TRAIN = 224
# Size of the image.jpg during test
_C.INPUT.SIZE_TEST = 224
# Minimum scale for the image.jpg during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image.jpg during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image.jpg horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image.jpg normalization
_C.INPUT.PIXEL_MEAN = [0.4914, 0.4822, 0.4465]
# Values to be used for image.jpg normalization
_C.INPUT.PIXEL_STD = [0.2023, 0.1994, 0.2010]
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT = '../data/dataset/cifar_100_dataset_file'
_C.DATASETS.AUGMENTATION = True
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2

# ---------------------------------------------------------------------------- #
# Optimizers
# ---------------------------------------------------------------------------- #
_C.OPT = CN()

_C.OPT.SGD = CN()

_C.OPT.SGD.OPTIMIZER_NAME = "SGD"
_C.OPT.SGD.LR = 0.1  # DEFAULT  0.001
_C.OPT.SGD.MOMENTUM = 0.9  # DEFAULT 0
_C.OPT.SGD.WEIGHT_DECAY = 0.0001  # DEFAULT 0
_C.OPT.SGD.DAMPENING = 0  # DEFAULT 0
_C.OPT.SGD.NESTEROV = True  # DEFAULT FALSE

# ------------------------------------------------------------------------------ #
_C.OPT.ADAM = CN()
_C.OPT.ADAM.OPTIMIZER_NAME = "ADAM"
_C.OPT.ADAM.LR = 0.001  # DEFAULT  0.001
_C.OPT.ADAM.BETAS = [0.9, 0.999]  # DEFAULT [0.9, 0.999]
_C.OPT.ADAM.EPS = 1e-08  # DEFAULT 1e-08
_C.OPT.ADAM.WEIGHT_DECAY = 1e-5  # DEFAULT 0
_C.OPT.ADAM.AMS_GRAD = False  # DEFAULT FALSE

# ------------------------------------------------------------------------------ #
_C.OPT.ADADELTA = CN()
_C.OPT.ADADELTA.OPTIMIZER_NAME = "ADADELTA"
_C.OPT.ADADELTA.LR = 0.0001  # DEFAULT  0.001
_C.OPT.ADADELTA.EPS = 1e-08  # DEFAULT 1e-08
_C.OPT.ADADELTA.WEIGHT_DECAY = 1e-5  # DEFAULT 0

# ------------------------------------------------------------------------------ #


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCHS = 10

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 400

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.WEIGHT = "/Model_CIFAR100_150.pth"

# ---------------------------------------------------------------------------- #
# Outputs
# ---------------------------------------------------------------------------- #
_C.DIR = CN()
_C.DIR.OUTPUT_DIR = "../outputs/first/check_pointers"
_C.DIR.TENSORBOARD_LOG = '../outputs/tensorboard_log'
_C.DIR.BEST_MODEL = '../outputs/best_models'
_C.DIR.FINAL_MODEL = '../outputs/final_model'
