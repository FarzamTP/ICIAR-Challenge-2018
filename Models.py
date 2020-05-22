from keras import layers
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2


class VGG:
    def __init__(self, model_name, input_shape, input_layer_name, num_classes):
        self.model_name = model_name
        self.input_shape = input_shape
        self.input_layer_name = input_layer_name
        self.num_classes = num_classes

    def construct_net(self):
        input_layer = layers.Input(shape=self.input_shape, name=self.input_layer_name)

        # Block 1
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1', kernel_regularizer=l2())(input_layer)
        # x = layers.Conv2D(64, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(3, 3), name='block1_pool')(x)

        # Block 2
        # x = layers.Conv2D(128, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1', kernel_regularizer=l2())(x)
        # x = layers.Conv2D(256, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block3_conv2')(x)
        # x = layers.Conv2D(256, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block3_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1', kernel_regularizer=l2())(x)
        # x = layers.Conv2D(512, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block4_conv2')(x)
        # x = layers.Conv2D(512, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block4_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1', kernel_regularizer=l2())(x)
        # x = layers.Conv2D(512, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block5_conv2')(x)
        # x = layers.Conv2D(512, (3, 3),
        #                   activation='relu',
        #                   padding='same',
        #                   name='block5_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu', name='fc2')(x)

        if self.num_classes == 4:
            x = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        else:
            x = layers.Dense(self.num_classes, activation='sigmoid', name='predictions')(x)

        model = Model(input=input_layer, output=x)
        return model

    def __str__(self):
        return self.model_name


class Resnet:
    def __init__(self, input_shape, input_layer_name, num_classes):
        self.input_shape = input_shape
        self.input_layer_name = input_layer_name
        self.num_classes = num_classes

    def construct_net(self):
        input_layer = layers.Input(shape=self.input_shape, name=self.input_layer_name)
        res_net = ResNet50(weights='imagenet', input_shape=self.input_shape, include_top=False)
        res_model = Model(input=input_layer, output=res_net.output)
        res_output = res_model.output

        x = layers.Flatten(name='flatten')(res_output)
        x = layers.Dense(128, activation='relu', name='fc2')(x)

        if self.num_classes == 4:
            x = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        else:
            x = layers.Dense(self.num_classes, activation='sigmoid', name='predictions')(x)

        model = Model(input=input_layer, output=x)
        return model
