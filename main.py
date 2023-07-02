import keras
import pandas as pd
import tensorflow_hub as hub
from matplotlib import pyplot as plt

#Parameters for variable learning rate with Keras optimizers.
iterationsToChangeAt = [500,600]
LRs = [.01, .001, 0.0001]
stepRate = keras.optimizers.schedules.PiecewiseConstantDecay(iterationsToChangeAt, LRs)

#Quick Image augmentation with ImageDataGenerator in Keras.
#ImageDataGenerator options: https://keras.io/api/preprocessing/image/#imagedatagenerator-class
dataGenerator = keras.preprocessing.image.ImageDataGenerator(validation_split=0.3, 
                                                             samplewise_center=True, 
                                                             fill_mode="nearest",
                                                             rotation_range=360,#Because we're using satellite data, let's allow 360 degree rotations.
                                                             brightness_range=[0.5,1.5],#Arbitrary range of brightness
                                                             horizontal_flip=True,#Randomly horizontal flip
                                                             vertical_flip=True #Random vertical flip
                                                             )

test = dataGenerator.flow_from_directory("./mercerImages", class_mode='categorical', 
                                            batch_size=32, subset="validation", 
                                            target_size=(224,224))
train = dataGenerator.flow_from_directory("./mercerImages", class_mode='categorical', 
                                        batch_size=32, subset="training", 
                                        target_size=(224,224))

#Transfer learning using resnet_50 and one Dense Layer. VGG16 is also a good choice for starting weights. Trainable=True so resnet weights are subject to change by our model.
model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
                   trainable=True),  
    keras.layers.Dense(21, activation='softmax')
])

#compiling with our variable learning rate
model.compile(optimizer=keras.optimizers.SGD(learning_rate=stepRate),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])


#Note that image augmentation will require more epochs to reasonably fit,
#as the amount of data going in is now higher!
modelHistory = model.fit(train, epochs=15, validation_data=test)

#quick viz just for fun
pd.DataFrame(modelHistory.history).plot(figsize=(8,5))
plt.show()
