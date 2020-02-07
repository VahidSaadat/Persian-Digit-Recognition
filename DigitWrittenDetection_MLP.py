from HodaDatasetReader import read_hoda_cdb
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from joblib import dump, load

# Reading images
print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')
print('Reading Test 20000.cdb ...')
test_images, test_labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')

# Preprocess
train_images_prep = []
for img in train_images:
    # Resize
    resize_img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
    # Threshold
    _, threshold_img = cv2.threshold(resize_img, 0, 1, cv2.THRESH_OTSU)
    # Feature extraction
    train_images_prep.append(threshold_img.flatten())
test_images_prep = []
for img in test_images:
    # Resize
    resize_img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
    # Threshold
    _, threshold_img = cv2.threshold(resize_img, 0, 1, cv2.THRESH_OTSU)
    # Feature extraction
    test_images_prep.append(threshold_img.flatten())

# MLP
clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(20, 20), activation='relu')

# split train-test validation
clf.fit(train_images_prep, train_labels)
pred = clf.predict(test_images_prep)
print(classification_report(test_labels, pred))