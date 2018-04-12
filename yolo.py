from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, BatchNormalization, Lambda, Activation, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
import keras
import numpy as np
from utils import WeightReader, decode_netout, draw_boxes, normalize
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import parse_annotation, BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
TRUE_BOX_BUFFER  = 50
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
CLASS_SCALE      = 1.0
WARM_UP_BATCHES  = 0
BATCH_SIZE       = 16

wt_path = 'yolov3.weights'
train_image_folder = '/hoya_src_root/train2014/'
train_annot_folder = '/hoya_src_root/train2014ann/'
valid_image_folder = '/hoya_src_root/val2014/'
valid_annot_folder = '/hoya_src_root/val2014ann/'

input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 3
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(32, (1,1), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 6
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 8
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 9
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 10
x = Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 13
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 14
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 15
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 17
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 19
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 21
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(x)
x = BatchNormalization(name='norm_21')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 22
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 23
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_23', use_bias=False)(x)
x = BatchNormalization(name='norm_23')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 24
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_24', use_bias=False)(x)
x = BatchNormalization(name='norm_24')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 25
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_25', use_bias=False)(x)
x = BatchNormalization(name='norm_25')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 26
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_26', use_bias=False)(x)
x = BatchNormalization(name='norm_26')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

skip1 = x

# Layer 27
x = Conv2D(512, (3,3), strides=(2,2), padding='same', name='conv_27', use_bias=False)(x)
x = BatchNormalization(name='norm_27')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 28
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_28', use_bias=False)(x)
x = BatchNormalization(name='norm_28')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 29
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_29', use_bias=False)(x)
x = BatchNormalization(name='norm_29')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 30
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_30', use_bias=False)(x)
x = BatchNormalization(name='norm_30')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 31
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_31', use_bias=False)(x)
x = BatchNormalization(name='norm_31')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 32
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_32', use_bias=False)(x)
x = BatchNormalization(name='norm_32')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 33
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_33', use_bias=False)(x)
x = BatchNormalization(name='norm_33')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 34
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_34', use_bias=False)(x)
x = BatchNormalization(name='norm_34')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 35
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_35', use_bias=False)(x)
x = BatchNormalization(name='norm_35')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 36
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_36', use_bias=False)(x)
x = BatchNormalization(name='norm_36')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 37
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_37', use_bias=False)(x)
x = BatchNormalization(name='norm_37')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 38
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_38', use_bias=False)(x)
x = BatchNormalization(name='norm_38')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 39
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_39', use_bias=False)(x)
x = BatchNormalization(name='norm_39')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 40
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_40', use_bias=False)(x)
x = BatchNormalization(name='norm_40')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 41
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_41', use_bias=False)(x)
x = BatchNormalization(name='norm_41')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 42
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_42', use_bias=False)(x)
x = BatchNormalization(name='norm_42')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 43
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_43', use_bias=False)(x)
x = BatchNormalization(name='norm_43')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

skip2 = x

# Layer 44
x = Conv2D(1024, (3,3), strides=(2,2), padding='same', name='conv_44', use_bias=False)(x)
x = BatchNormalization(name='norm_44')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 45
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_45', use_bias=False)(x)
x = BatchNormalization(name='norm_45')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 46
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_46', use_bias=False)(x)
x = BatchNormalization(name='norm_46')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 47
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_47', use_bias=False)(x)
x = BatchNormalization(name='norm_47')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 48
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_48', use_bias=False)(x)
x = BatchNormalization(name='norm_48')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 49
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_49', use_bias=False)(x)
x = BatchNormalization(name='norm_49')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 50
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_50', use_bias=False)(x)
x = BatchNormalization(name='norm_50')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 51
shortcut = x
#shortcut = BatchNormalization()(shortcut)
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_51', use_bias=False)(x)
x = BatchNormalization(name='norm_51')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 52
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_52', use_bias=False)(x)
x = BatchNormalization(name='norm_52')(x)
x = LeakyReLU(alpha=0.1)(x)
x = keras.layers.add([x, shortcut])
x = Activation('linear')(x)

# Layer 53
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_53', use_bias=False)(x)
x = BatchNormalization(name='norm_53')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 54
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_54', use_bias=False)(x)
x = BatchNormalization(name='norm_54')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 55
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_55', use_bias=False)(x)
x = BatchNormalization(name='norm_55')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 56
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_56', use_bias=False)(x)
x = BatchNormalization(name='norm_56')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 57
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_57', use_bias=False)(x)
x = BatchNormalization(name='norm_57')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 58
yolov3_1 = x
yolov3_1 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_58', use_bias=False)(yolov3_1)
yolov3_1 = BatchNormalization(name='norm_58')(yolov3_1)
yolov3_1 = LeakyReLU(alpha=0.1)(yolov3_1)

# Layer 59
yolov3_1 = Conv2D(255, (1,1), strides=(1,1), padding='same', name='conv_59', use_bias=True)(yolov3_1)
#yolov3_1 = Activation('linear')(yolov3_1)

yolov3_1 = Reshape((13, 13, 3, 85))(yolov3_1)

# Layer 60
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_60', use_bias=False)(x)
x = BatchNormalization(name='norm_60')(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling2D(2)(x)
x = keras.layers.concatenate([x, skip2])

# Layer 61
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_61', use_bias=False)(x)
x = BatchNormalization(name='norm_61')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 62
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_62', use_bias=False)(x)
x = BatchNormalization(name='norm_62')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 63
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_63', use_bias=False)(x)
x = BatchNormalization(name='norm_63')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 64
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_64', use_bias=False)(x)
x = BatchNormalization(name='norm_64')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 65
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_65', use_bias=False)(x)
x = BatchNormalization(name='norm_65')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 66
yolov3_2 = x
yolov3_2 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_66', use_bias=False)(yolov3_2)
yolov3_2 = BatchNormalization(name='norm_66')(yolov3_2)
yolov3_2 = LeakyReLU(alpha=0.1)(yolov3_2)

# Layer 67
yolov3_2 = Conv2D(255, (1,1), strides=(1,1), padding='same', name='conv_67', use_bias=True)(yolov3_2)
#yolov3_2 = Activation('linear')(yolov3_2)

yolov3_2 = Reshape((26, 26, 3, 85))(yolov3_2)

# Layer 68
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_68', use_bias=False)(x)
x = BatchNormalization(name='norm_68')(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling2D(2)(x)
x = keras.layers.concatenate([x, skip1])

# Layer 69
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_69', use_bias=False)(x)
x = BatchNormalization(name='norm_69')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 70
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_70', use_bias=False)(x)
x = BatchNormalization(name='norm_70')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 71
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_71', use_bias=False)(x)
x = BatchNormalization(name='norm_71')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 72
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_72', use_bias=False)(x)
x = BatchNormalization(name='norm_72')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 73
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_73', use_bias=False)(x)
x = BatchNormalization(name='norm_73')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 74
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_74', use_bias=False)(x)
x = BatchNormalization(name='norm_74')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 75
x = Conv2D(255, (1,1), strides=(1,1), padding='same', name='conv_75', use_bias=True)(x)
#x = Activation('linear')(x)

x = Reshape((52, 52, 3, 85))(x)

# feature_extractor
model = Model(input_image,[yolov3_1,yolov3_2,x], name="yolov3")

print(model.summary())

weight_reader = WeightReader(wt_path)

weight_reader.reset()

nb_conv = 75

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))

    if i < nb_conv and i not in (59,67):
        norm_layer = model.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])

model.save('yolo.h5')