#!/usr/bin/env python

##############
#### Your name:
##############

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog
import skimage
import ransac_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import SGDClassifier

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################

        # Please do not modify the return type below
        # pre_hor = filters.prewitt_h(data)
        # pre_ver = filters.prewitt_v(data) 
        # ed_sobel = filters.sobel(data)
        # hist, hist_centers = exposure.histogram(data)
        grayify = RGB2GrayTransformer()
        hogify = HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2,2), 
            orientations=9, 
            block_norm='L2-Hys'
        )
        scalify = StandardScaler()
 
        # call fit_transform on each transform converting X_train step by step
        X_train_gray = grayify.fit_transform(data)
        X_train_hog = hogify.fit_transform(X_train_gray)
        feature_data = scalify.fit_transform(X_train_hog)

        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        
        ########################
        ######## YOUR CODE HERE
        ########################
        sgd_clf = SGDClassifier(random_state=1, max_iter=2000, tol=1e-7)
        sgd_clf.fit(train_data, train_labels)
        self.classifier = sgd_clf

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        ########################

        # Please do not modify the return type below

        predicted_labels = self.classifier.predict(data)
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        ######## YOUR CODE HERE
        ########################

        # Please do not modify the return type below
        return slope, intercept
        
def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()

 
