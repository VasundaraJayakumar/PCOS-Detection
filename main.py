import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
import seaborn as sns
from sklearn import preprocessing
import joblib
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

SIZE = 256
SEED_TRAINING = 121
SEED_TESTING = 197
SEED_VALIDATION = 164
CHANNELS = 3
n_classes = 3
EPOCHS = 50
BATCH_SIZE = 16
input_shape = (SIZE, SIZE, CHANNELS)

def training(path):
    train_images = []
    train_labels = []

    path = os.path.join(path, '*')
    for directory_path in glob.glob(path):
        label = os.path.basename(directory_path)

        for img_path in glob.glob(os.path.join(directory_path, '*.JPG')):
            img = cv2.imread(img_path)
            if img is not None and not img.size == 0:
                img = cv2.resize(img, (SIZE, SIZE))

                train_images.append(img)
                train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images = train_images / 255.0
    return train_images, train_labels

def testing(path):
    test_images = []
    test_labels = []

    path = os.path.join(path, '*')
    for directory_path in glob.glob(path):
        label = os.path.basename(directory_path)

        for img_path in glob.glob(os.path.join(directory_path, '*.JPG')):
            img = cv2.imread(img_path)
            if img is not None and not img.size == 0:
                img = cv2.resize(img, (SIZE, SIZE))

                test_images.append(img)
                test_labels.append(label)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    test_images = test_images / 255.0
    return test_images, test_labels

#X_test, y_test_labels = training(r'C:\Users\VASUNDARA\Desktop\DS CODE\PCOS-Detection\data\test')
#X_train, y_train_labels = training(r'C:\Users\VASUNDARA\Desktop\DS CODE\PCOS-Detection\data\train')

le = preprocessing.LabelEncoder()
le.fit(y_train_labels)
train_label_encoded = le.transform(y_train_labels)
test_label_encoded = le.transform(y_test_labels)

y_train, y_test = train_label_encoded, test_label_encoded

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in vgg_model.layers:
    layer.trainable = False

vgg_model.summary()

feature_extractor = vgg_model.predict(X_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_train_features = features

feature_extractor_test = vgg_model.predict(X_test)
features_test = feature_extractor_test.reshape(feature_extractor_test.shape[0], -1)
X_test_features = features_test

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': n_classes,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

train_data = lgb.Dataset(X_train_features, label=y_train)
test_data = lgb.Dataset(X_test_features, label=y_test)

model = lgb.train(params, train_data, valid_sets=[test_data])

prediction = np.argmax(model.predict(X_test_features), axis=1)
prediction = le.inverse_transform(prediction)

print('Accuracy : ', metrics.accuracy_score(y_test_labels, prediction))

cm = confusion_matrix(y_test_labels, prediction)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True)

print(classification_report(y_test_labels, prediction))

joblib.dump(model, 'xray_lightgbm.pkl')
