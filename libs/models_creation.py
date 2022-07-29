from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Input
from tensorflow.keras.losses import CategoricalCrossentropy


class ResNet_Model(Model):
    def __init__(self, num_classes, input_shape, layers_num):
        super().__init__()
        downloaded_model = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=input_shape))
        model = Sequential()
        model.add(downloaded_model)
        model.add(Flatten(name="flatten"))
        for _ in range(layers_num-1):
          model.add(Dense(256, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        self.model = model
        print(self.model.summary())

    def __call__(self, inp, training, mask=None):
      # inp, tar = inputs
      output = self.model(inp, training, mask=mask)

      return output
  

class VGG16_Model(Model):
    def __init__(self, num_classes, input_shape, layers):
        super().__init__()
        downloaded_model = VGG16(weights=None, input_tensor=Input(shape=input_shape),include_top=False)
        model = Sequential()
        model.add(downloaded_model)
        model.add(Flatten(name="flatten"))
        for _ in range(layers-1):
            model.add(Dense(256, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        self.model = model
        print(self.model.summary())

    def __call__(self, inp, training, mask=None):
      # inp, tar = inputs
      output = self.model(inp, training, mask=mask)

      return output

class SimpleDNN_Model(Model):
  def __init__(self,input_shape, num_layers, num_classes):
    super().__init__()
    self.shape = input_shape
    self.num_layers = num_layers
    self.num_classes = num_classes

  def create_model(self):
    input = Input(shape=self.shape)
    output = Flatten(input_shape=(244,244,3))(input)
    for _ in range(self.num_layers-1):
      output = Dense(256, activation="relu")(output)
    output = Dense(self.num_classes, activation="softmax")(output)

    self.model = Model(inputs=input, outputs=output)
    loss = CategoricalCrossentropy(
              from_logits=False,
              label_smoothing=0.0,
              axis=-1,
              name='categorical_crossentropy'
            )
    self.model.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])
    return self.model
