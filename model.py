import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras import optimizers
from keras.callbacks import ModelCheckpoint

samples = np.empty([0])


class Sample():
    def __init__(self, image_path, angle, flip):
        self.image_path = image_path
        self.angle = angle
        self.flip = flip


def extractFileName(path):
    path = path.replace('\\', '/')
    return path.split('/')[-1]


def importCsv(path, negativeOnly=None, positiveOnly=None, curvesOnly=None):
    result = []
    lines = []
    with open('./data/' + path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (len(line) > 1):
                lines.append(line)

    print(path, len(lines), 'records')

    for line in lines:
        centerImage = './data/' + path + '/IMG/' + extractFileName(line[0])
        leftImage = './data/' + path + '/IMG/' + extractFileName(line[1])
        rightImage = './data/' + path + '/IMG/' + extractFileName(line[2])
        measurement = float(line[3])

        if abs(measurement) > 0.9:
            continue

        if negativeOnly is not None and measurement >= 0:
            continue

        if positiveOnly is not None and measurement <= 0:
            continue

        if curvesOnly is not None and abs(measurement) < 0.01:
            continue

        camera_correction = 0.25

        result.append(Sample(centerImage, measurement, False))
        result.append(Sample(leftImage, measurement + camera_correction, False))
        result.append(Sample(rightImage, measurement - camera_correction, False))

        result.append(Sample(centerImage, measurement, True))
        result.append(Sample(leftImage, measurement + camera_correction, True))
        result.append(Sample(rightImage, measurement - camera_correction, True))

    return result


# Drive in the middle of the road
samples = np.append(samples, importCsv('ccw'))
samples = np.append(samples, importCsv('cw'))

# Curves
samples = np.append(samples, importCsv('curves-ccw', curvesOnly = True ))
samples = np.append(samples, importCsv('curves-2', curvesOnly = True))
samples = np.append(samples, importCsv('curves-3', curvesOnly = True))
samples = np.append(samples, importCsv('curves-4'))
samples = np.append(samples, importCsv('curves-5'))
samples = np.append(samples, importCsv('curves-6'))

# Recovery
samples = np.append(samples, importCsv('recovery-minus', negativeOnly = True))
samples = np.append(samples, importCsv('recovery-minus-2', negativeOnly = True))
samples = np.append(samples, importCsv('bridge-2-minus', negativeOnly = True))
samples = np.append(samples, importCsv('recovery-plus', positiveOnly = True))
samples = np.append(samples, importCsv('recovery-plus-2', positiveOnly = True))
samples = np.append(samples, importCsv('bridge-3-plus', positiveOnly = True))
np.random.shuffle(samples)

print('Total samples: ', len(samples))

# Preprocess images

def preprocess(sample):
    image = cv2.imread(sample.image_path)
    angle = sample.angle

    if (sample.flip):
        image = cv2.flip(image, 1)
        angle = -angle

    # Crop
    image = image[60:140]
    return image, angle


def preprocess_color(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    gray = image[:, :, 1]
    return np.dstack((gray, hls[:, :, 2]))


train_samples, valid_samples = train_test_split(samples, test_size=0.2)

ch, row, col = 2, 80, 320  # Trimmed image format

print('Train samples: ', len(train_samples))
print('Validation samples: ', len(valid_samples))


# Create generators

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                image, angle = preprocess(sample)
                image = preprocess_color(image)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(valid_samples, batch_size=32)

# Create model

alpha = 0.05

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.0,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch), name='input'))

model.add(Conv2D(24, 5, 5, name='conv_1_5x5', subsample=(2,3)))
model.add(BatchNormalization(name='norm_1'))
model.add(LeakyReLU(alpha=alpha, name='act_1'))

model.add(Conv2D(36, 5, 5, name='conv_2_5x5', subsample=(2,2)))
model.add(BatchNormalization(name='norm_2'))
model.add(LeakyReLU(alpha=alpha, name='act_2'))

model.add(Conv2D(48, 5, 5, border_mode='valid', name='conv3_5x5', subsample=(2,2)))
model.add(BatchNormalization(name='norm_3'))
model.add(LeakyReLU(alpha=alpha, name='act_3'))

model.add(Conv2D(64, 3, 3, border_mode='valid', name='conv_4_3x3', subsample=(2,1)))
model.add(BatchNormalization(name='norm_4'))
model.add(LeakyReLU(alpha=alpha, name='act_4'))

model.add(Conv2D(64, 3, 3, border_mode='valid', name='conv_5_3x3'))
model.add(BatchNormalization(name='norm_5'))
model.add(LeakyReLU(alpha=alpha, name='act_5'))

model.add(Flatten(name='flat'))
model.add(Dropout(0.5, name='drop_1'))
model.add(Dense(128, activation='elu', name='fc_1'))
model.add(Dropout(0.5, name='drop_2'))
model.add(Dense(64, activation='elu', name='fc_2'))
model.add(Dense(16, activation='elu', name='fc_3'))
model.add(Dense(1, name='regression'))
model.add(Activation(activation='tanh', name='result'))

print(model.summary())


# Train and save the model

NUM_EPOCHS = 15

optimizer = optimizers.Adam(lr=0.0005)


checkpoint = ModelCheckpoint('best-model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

model.compile(loss='mse', optimizer=optimizer)

history_object = model.fit_generator(
    train_generator,
    samples_per_epoch = len(train_samples),
    validation_data = validation_generator,
    nb_val_samples = len(valid_samples),
    nb_epoch=NUM_EPOCHS,
    verbose = 1,
    callbacks=callbacks_list
)

model.save('model.h5')
print('model saved')

