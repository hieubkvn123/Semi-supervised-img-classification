from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras import regularizers

def conv_block(input_shape, filters, kernel_size=3, pooling=False, batchnorm=True):
    inputs = Input(shape=input_shape)
    output = Conv2D(filters, kernel_size=kernel_size, padding='same')(inputs)
    output = LeakyReLU(alpha=0.2)(output)
    
    if(batchnorm):
        output = BatchNormalization()(output)

    if(pooling):
        output = MaxPooling2D(pool_size=(2,2))(output)

    block = Model(inputs = inputs, outputs = output)
    return block

def build_cls_model(input_dim, n_classes=2, n_conv_blocks=2, logits_dim=128):
    '''
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
    '''
    
    inputs = Input(shape=input_dim)
    output = inputs

    for i in range(n_conv_blocks):
        filters = 8 * (2 ** i)
        input_shape = output.shape[1:]
        output = conv_block(input_shape, filters=filters, kernel_size=5, batchnorm=False, pooling=True)(output)

    
    output = Flatten()(output)
    logits = Dense(logits_dim)(output)
    prob = Dense(n_classes, activation='sigmoid')(logits)

    model = Model(inputs=inputs, outputs=[prob, logits], name="CLS_MODEL")

    return model
    
