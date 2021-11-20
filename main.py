from PIL import Image
import numpy
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 4
input_shape = (256, 256, 3)
target_size = (256, 256)

def load_data(folder):
    label_map = {'apple': [1, 0, 0, 0], 'banana': [0, 1, 0, 0], 'orange': [0, 0, 1, 0], 'mixed': [0, 0, 0, 1]}

    x_list = []
    y_list = []
    for file_name in listdir(folder): # list all the file in this folder, file_name(string)
        if file_name.endswith('.jpg'): # filename.jpg
            label_name = file_name.split('_')[0] # file_name.split('_') => [appple, 1.jpg] => label_name: apple

            file_path = join(folder, file_name) # file_path: ./train/apple_1.jpg
            image = Image.open(file_path)
            channel_size = numpy.asarray(image).shape[-1] # numpy.asarray(image).shape => (height, width, dimension)
            background = image
            if channel_size == 4: # RGBA => RGB
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])

            if channel_size == 4 or channel_size == 3:
                background = background.resize(target_size) # resize to target_size
                data = numpy.asarray(background) # shape (256, 256, 3)
                x_list.append(data)
                y_list.append(label_map[label_name])

    return numpy.array(x_list, dtype='float32'), numpy.array(y_list, dtype='uint8')
    # shape: (size, height, width, 3), (size, 4)


if __name__ == '__main__':
    x_train, y_train = load_data('./train') # read file
    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)

    x_test, y_test = load_data('./test')
    print('x_test: ', x_test.shape)
    print('y_test: ', y_test.shape)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 16
    epochs = 15

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
