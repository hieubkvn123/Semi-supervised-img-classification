from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras import regularizers

def build_cls_model(input_dim, backbone='resnet', n_classes=2, logits_dim=32):
    if(backbone not in ["resnet", "inception", "vgg16"]):
        raise Exception("Invalid backbone")

    if(backbone == "resnet"):
        backbone = applications.resnet.ResNet152(include_top=False, weights=None)
    elif(backbone == "inception"):
        backbone = applications.inception_v3.InceptionV3(include_top=False, weights=None)
    elif(backbone == "vgg16"):
        backbone = applications.vgg16.VGG16(include_top=False, weights=None)

    inputs = Input(shape=input_dim)
    output = backbone(inputs)
    output = AveragePooling2D(pool_size=(2, 2))(output)
    output = Flatten()(output)
    
    logits = Dense(logits_dim)(output)
    prob = Dense(n_classes, activation='softmax')(logits)

    model = Model(inputs=inputs, outputs=[prob, logits], name="CLS_MODEL")

    return model
    
