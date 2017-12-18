import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import time
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.layers import Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from scipy import misc
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
import cv2


def get_images(df):
    '''Create 3-channel 'images'. Return rescale-normalised images.'''
    images = []
    for i, row in df.iterrows():
        # Formulate the bands as 75x75 arrays
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        # Rescale
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        images.append(rgb)
    return np.array(images)


def ConvBlock(model, layers, filters):
    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def create_model():
    '''Create the FCN and return a keras model.'''

    model = Sequential()

    # Input image: 75x75x3
    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
    ConvBlock(model, 1, 32)
    # 37x37x32
    ConvBlock(model, 1, 64)
    # 18x18x64
    ConvBlock(model, 1, 128)
    # 9x9x128
    ConvBlock(model, 1, 128)
    # 4x4x128
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(2, (3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    # 4x4x2
    model.add(Activation('softmax'))

    return model


def get_cm(inp, label):
    '''Convert the 4x4 layer data to a 75x75 image.'''
    conv = np.rollaxis(conv_fn([inp, 0])[0][0], 2, 0)[label]
    return scipy.misc.imresize(conv, (75, 75), interp='nearest')


def info_img(im_idx, model):
    '''Generate heat maps for the boat (boatness) and iceberg (bergness) for image im_idx.'''
    if (yv[im_idx][1] == 1.0):
        img_type = 'iceberg'
    else:
        img_type = 'boat'
    inp = np.expand_dims(Xv[im_idx], 0)
    img_guess = np.round(model.predict(inp)[0], 2)
    if (img_guess[1] > 0.5):
        guess_type = 'iceberg'
    else:
        guess_type = 'boat'
    cm0 = get_cm(inp, 0)
    cm1 = get_cm(inp, 1)
    print('truth: {}'.format(img_type))
    print('guess: {}, prob: {}'.format(guess_type, img_guess))
    plt.figure(1, figsize=(10, 10))
    plt.subplot(121)
    plt.title('Boatness')
    plt.imshow(Xv[im_idx])
    plt.imshow(cm0, cmap="cool", alpha=0.5)
    plt.subplot(122)
    plt.title('Bergness')
    plt.imshow(Xv[im_idx])
    plt.imshow(cm1, cmap="cool", alpha=0.5)


def load_model(date_pattern):
    json_file = open("models/model_{0}.json".format(date_pattern), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model_w_{0}.h5".format(date_pattern))
    loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    print("Loaded model from disk")

    return loaded_model


def store_model(model, name=None):
    model_json = model.to_json()
    now_date = time.strftime("%H_%M_%S")

    if name is not None:
        now_date += name

    with open("models/model_{0}.json".format(now_date), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("models/model_w_{0}.h5".format(now_date))
    print("Saved model to disk")


def train_store_model():
    custom_model = create_model()
    custom_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    print(custom_model.summary())
    init_epo = 0
    num_epo = 30
    end_epo = init_epo + num_epo
    print('lr = {}'.format(K.get_value(custom_model.optimizer.lr)))
    history = custom_model.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=32, epochs=end_epo,
                               initial_epoch=init_epo)
    store_model(custom_model)

    return custom_model


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def getModel15():
    # Build keras model

    model = Sequential()

    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


train = pd.read_json('input/train.json')
X = get_images(train)
# y = to_categorical(train.is_iceberg.values, num_classes=2)

# CNN LB 1.5
y = np.array(train['is_iceberg'])
train.inc_angle = train.inc_angle.replace('na', 0)
idx_tr = np.where(train.inc_angle > 0)
y = y[idx_tr[0]]
X = X[idx_tr[0], ...]
Xtr_more = get_more_images(X)
Ytr_more = np.concatenate((y, y, y))

model = getModel15()
model.summary()
batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath='.mdl_wts.hdf5')

score = model.evaluate(X, y, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

df_test = pd.read_json('input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na', 0)
Xtest = (get_images(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('submission.csv', index=False)

# Xtr, Xv, ytr, yv = train_test_split(X, y, shuffle=False, test_size=0.20)

# model = train_store_model()
# model = load_model('23_03_51')
from se_resnet import SEResNet

# model = SEResNet(input_shape=(75, 75, 3), classes=2)
# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
# print(model.summary())


# init_epo = 0
# num_epo = 30
# end_epo = init_epo + num_epo
# print('lr = {}'.format(K.get_value(model.optimizer.lr)))
# history = model.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=32, epochs=end_epo,
#                     initial_epoch=init_epo)

# store_model(model, name='CNN1DOT5')

# l = model.layers
# conv_fn = K.function([l[0].input, K.learning_phase()], [l[-4].output])

# info_img(13, model)

# model = load_model("22_49_25SERESNET1")

# test = pd.read_json('input/test.json')
# Xtest = get_images(test)
# test_predictions = model.predict(Xtest)
# submission = pd.DataFrame({'id': test['id'], 'is_iceberg': test_predictions[:, 1]})
# submission.to_csv('sub_fcn_RESNET1_1.csv', index=False)
