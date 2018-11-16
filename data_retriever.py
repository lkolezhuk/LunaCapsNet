import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import os
import multiprocessing
from scipy import misc
from scipy import ndimage
import random
from sklearn.utils import shuffle
from pathlib import Path
from utils import hist_match, image_preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import cv2
import conf
from tempfile import TemporaryFile

class DataRetriever:

    def __init__(self, testing=False):
        self.data = []

        self.__testing = testing

        self.__classSplit = conf.CLASS_SPLIT
        self.__posClassSamples = 0
        self.__negClassSamples = 0

        # self.__trainvalRefPath = os.path.join(DATASET_PATH, DATA_TRAINVAL_FILENAME)
        # self.__testRefPath = os.path.join(DATASET_PATH, DATA_TESTING_FILENAME)
        #
        # self.__labelsPath = os.path.join(DATASET_PATH, DATA_LABELS_FILENAME)

        self.__labelsTrainValPath = os.path.join(conf.DATASET_PATH, conf.DATA_LABELS_TRAINVAL_FILENAME)
        self.__labelsTestingPath = os.path.join(conf.DATASET_PATH, conf.DATA_LABELS_TESTING_FILENAME)
        self.__imagesFolderPath = os.path.join(conf.DATASET_PATH, "images/")
        self.__checkFiles()

    def load(self, size=None):
        if(os.path.isfile('dataset-images.npy') and os.path.isfile('dataset-labels.npy')):
            print('Prestored dataset files found')
            images_rest = np.load('dataset-images.npy')
            labels_rest = np.load('dataset-labels.npy')
            self.data = (images_rest, labels_rest)
            print('Loaded {0} samples from prestored file'.format(str(len(images_rest))))
        else:
            # Loads data
            label_entities = self.__getLocationsFromCSV()

            (pos_entities, neg_entities) = self.__splitLocations(label_entities)
            self.data = self.__getImagesAndLabels(size, pos_entities, neg_entities)

            self.__posClassSamples = len(pos_entities)
            self.__negClassSamples = len(neg_entities)
            self.__store()
            self.__log()
    def getTrainValSplit(self, validationPercentage):
        return train_test_split(self.data[0], self.data[1], test_size=validationPercentage)
    def getTest(self):
        return self.data[0], self.data[1]

    def __checkFiles(self):
        tvLabelsFile = Path(self.__labelsTrainValPath)
        if not tvLabelsFile.is_file():
            raise Exception("Could not find file " + str(self.__labelsTrainValPath))

        testLabelsFile = Path(self.__labelsTestingPath)
        if not testLabelsFile.is_file():
            raise Exception("Could not find file " + str(self.__labelsTestingPath))

        imagesDir = Path(self.__imagesFolderPath)
        if not imagesDir.is_dir():
            raise Exception("Could not find input directory at " + str(self.__imagesFolderPath))
        # trainvalRefFile = Path(self.__trainvalRefPath)
        # if not trainvalRefFile.is_file():
        #     raise FileNotFoundError("Could not find file " + str(self.__trainvalRefPath))
        #
        # testRefFile = Path(self.__testRefPath)
        # if not testRefFile.is_file():
        #     raise FileNotFoundError("Could not find file " + str(self.__testRefPath))
        #
        # labelsFile = Path(self.__labelsPath)
        # if not labelsFile.is_file():
        #     raise FileNotFoundError("Could not find file " + str(self.__labelsPath))
    def __store(self):
        print('Storing dataset to a file')
        assert self.data is not None and len(self.data) > 0

        np.save('dataset-images.npy', self.data[0])
        np.save('dataset-labels.npy', self.data[1])
        print('Successfully stored dataset')

    def __getLocationsFromCSVSplitInitialDataSet(self):
        locations = []
        with open(self.__labelsPath) as csv_label_file:
            reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
            label_entities = list(reader)
            if not self.__testing: # Training and validation mode
                with open(self.__trainvalRefPath) as ref_file:
                        mode_entities = ref_file.read().splitlines()
                        for m_entry in mode_entities:
                            label_entity = list(filter(lambda x: x[0] == m_entry, label_entities))[0]
                            locations.append(label_entity)
                            print(m_entry)
                        with open(conf.DATA_LABELS_TRAINVAL_FILENAME, 'w', newline='') as resultFile:
                            wr = csv.writer(resultFile, dialect='excel')
                            wr.writerows(locations)
            else: # Testing mode
                with open(self.__testRefPath) as ref_file:
                    mode_entities = ref_file.read().splitlines()
                    for m_entry in mode_entities:
                        label_entity = list(filter(lambda x: x[0] == m_entry, label_entities))[0]
                        locations.append(label_entity)
                        print(m_entry)
                    with open(conf.DATA_LABELS_TESTING_FILENAME, 'w', newline='') as resultFile:
                        wr = csv.writer(resultFile, dialect='excel')
                        wr.writerows(locations)
    def __getLocationsFromCSV(self):
        csv_label_file_path = self.__labelsTrainValPath
        if self.__testing:
            csv_label_file_path = self.__labelsTestingPath

        with open(csv_label_file_path) as csv_label_file:
            reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
            label_entities = list(reader)
        return label_entities
    def __splitLocations(self, label_entities):
        pos_entities = []
        neg_entities = []
        assert len(label_entities) > 0
        for entity in label_entities:
            # if self.__encode_chestx_label_unique(entity[1], "Mass", "No Finding") == 1:
            #     pos_entities.append(entity)
            # elif self.__encode_chestx_label_unique(entity[1], "Mass", "No Finding") == 0:
            #     neg_entities.append(entity)
            label_enc = self.__encode_chestx_label(entity[1], conf.POSITIVE_LABELS, conf.NEGATIVE_LABELS)
            if label_enc is not None and label_enc > 0:
                pos_entities.append(entity)
            elif label_enc == 0:
                neg_entities.append(entity)
        return (pos_entities, neg_entities)

    def __getImagesAndLabels(self, data_size, pos_locations, neg_locations):

        assert len(pos_locations) > 0
        assert len(neg_locations) > 0

        filename = os.path.join(self.__imagesFolderPath, "00000003_006.png")
        hist_template = ndimage.imread(filename, True, 'L')
        if data_size is None:
            pos_indexes = range(0, len(pos_locations))
            neg_indexes = range(0, len(neg_locations))
        else:
            pos_indexes = random.sample(range(1, len(pos_locations)), int(data_size * self.__classSplit))
            neg_indexes = random.sample(range(1, len(neg_locations)), int(data_size * (1. - self.__classSplit)))

        pos_locations_selected = []
        for ind in pos_indexes:
            pos_locations_selected.append(pos_locations[ind])

        neg_locations_selected = []
        for ind in neg_indexes:
            neg_locations_selected.append(neg_locations[ind])

        pos_locations = pos_locations_selected
        neg_locations = neg_locations_selected

        images = []
        labels = []
        total_files_to_read = len(pos_locations) + len(neg_locations)
        for ind, label_enry in enumerate(pos_locations):
            filename = os.path.join(self.__imagesFolderPath, label_enry[0])
            try:
                # im = ndimage.imread(filename, True, 'L')
                im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                im = hist_match(im, hist_template)

                im = misc.imresize(im, (conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM), 'bilinear')
                im = image_preprocessing(im)
                im = im.reshape(conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1)

                images.append(im)
                labels.append(self.__encode_chestx_label(label_enry[1], conf.POSITIVE_LABELS, conf.NEGATIVE_LABELS))
                printProgressBar(ind+1, total_files_to_read , prefix='Loaded files:', suffix='Complete', length=50)
            except (FileNotFoundError, OSError, AttributeError):
                print('Missing {0}'.format(str(filename)))
            except ValueError as e:
                print('Value error: ' + str(e))

        for ind, label_enry in enumerate(neg_locations):
            filename = os.path.join(self.__imagesFolderPath, label_enry[0])
            try:
                # im = ndimage.imread(filename, True, 'L')
                im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                im = hist_match(im, hist_template)

                im = misc.imresize(im, (conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM), 'bilinear')
                im = image_preprocessing(im)
                im = im.reshape(conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1)

                images.append(im)
                labels.append(0)
                printProgressBar(len(pos_locations) + ind + 1, total_files_to_read, prefix='Loaded files:', suffix='Complete', length=50)
            except (FileNotFoundError, OSError, AttributeError):
                print('Missing {0}'.format(str(filename)))
            except ValueError as e:
                print('Value error: ' + str(e))
        # locations = pos_locations + neg_locations

        images, labels = shuffle(images, labels)

        x = np.asarray(images)
        x = x.reshape(-1, conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1).astype('float32')

        y = np.asarray(labels)
        y = to_categorical(y, conf.NUM_CLASSES)
        return x, y

    # def __getImagesAndLabelsMT(self, data_size, pos_locations, neg_locations):
    #
    #     assert len(pos_locations) > 0
    #     assert len(neg_locations) > 0
    #
    #     filename = os.path.join(self.__imagesFolderPath, "00000003_006.png")
    #     hist_template = ndimage.imread(filename, True, 'L')
    #     if data_size is None:
    #         pos_indexes = range(0, len(pos_locations))
    #         neg_indexes = range(0, len(neg_locations))
    #     else:
    #         pos_indexes = random.sample(range(1, len(pos_locations)), int(data_size * self.__classSplit))
    #         neg_indexes = random.sample(range(1, len(neg_locations)), int(data_size * (1. - self.__classSplit)))
    #
    #     pos_locations_selected = []
    #     for ind in pos_indexes:
    #         pos_locations_selected.append(pos_locations[ind])
    #
    #     neg_locations_selected = []
    #     for ind in neg_indexes:
    #         neg_locations_selected.append(neg_locations[ind])
    #
    #     pos_locations = pos_locations_selected
    #     neg_locations = neg_locations_selected
    #
    #     images = []
    #     labels = []
    #     total_files_to_read = len(pos_locations) + len(neg_locations)
    #     processed_images = 0
    #
    #     pool = multiprocessing.Pool(4)
    #     (images, labels) = pool.map(hui, pos_locations)
    #     print('hui')
    #     images, labels = shuffle(images, labels)
    #
    #     x = np.asarray(images)
    #     x = x.reshape(-1, conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1).astype('float32')
    #
    #     y = np.asarray(labels)
    #     y = to_categorical(y, conf.NUM_CLASSES)
    #     return x, y



    def __encode_chestx_label(self, label, positive_labels, negative_labels):
        if label in positive_labels and label.find('|') == -1:
            return positive_labels.index(label) + 1
        elif label in negative_labels:
            return 0
    def __encode_chestx_label_unique(self, label, positive_label, negative_label):
        if label == positive_label:
            return 1
        elif label == negative_label:
            return 0
    def __encode_chestx_labels(self, labels, positive_labels, negative_labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.__encode_chestx_label(label, positive_labels, negative_labels))
        return encoded_labels

    def __log(self):
        print("Loaded data")
        print("Testing mode: {0}".format(str(self.__testing)))
        print("Size: {0}".format(len(self.data)))
        print("Class split: {0}".format(str(self.__classSplit)))
        if self.__testing:
            print("Class split ignored due to testing mode. Considering initial split instead")
            print("Used class split: {0}".format(str(float(self.__posClassSamples)/self.__negClassSamples)))
        else:
            print("Loaded {0} samples".format(str(conf.DATASET_SAMPLES)))
        print("POS class: {0} samples. Negative class: {1} samples ".format(self.__posClassSamples, self.__negClassSamples))
    def test(self):
        lenc = self.__encode_chestx_label("Mass", ["Mass", "Pigeon", "Tree"], ["No Finding"])
        lenc1 = self.__encode_chestx_label("Pigeon", ["Mass", "Pigeon", "Tree"], ["No Finding"])
        lenc2 = self.__encode_chestx_label("No Finding", ["Mass", "Pigeon", "Tree"], ["No Finding"])
        lencc = self.__encode_chestx_labels(["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia", "No Finding"], conf.POSITIVE_LABELS, conf.NEGATIVE_LABELS)
        i = 11

def load_image_and_label(data, hist_template, total_files_to_read, ind, folderPath):
        label_enry = data
        filename = os.path.join(folderPath, label_enry[0])
        try:
            # im = ndimage.imread(filename, True, 'L')
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            im = hist_match(im, hist_template)

            im = misc.imresize(im, (conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM), 'bilinear')
            im = image_preprocessing(im)
            im = im.reshape(conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1)

            printProgressBar(ind + 1, total_files_to_read, prefix='Loaded files:', suffix='Complete', length=50)
            return im

        except (FileNotFoundError, OSError, AttributeError):
            print('Missing {0}'.format(str(filename)))
        except ValueError as e:
            print('Value error: ' + str(e))
def hui(x):
        print('hui')

if __name__=="__main__":
    dataset = DataRetriever()
    dataset.load(500)
    (images, labels) = dataset.getTest()




    images_rest = np.load('dataset-images.npy')
    labels_rest = np.load('dataset-labels.npy')
    # dataTesting = DataRetriever(testing=True)
    # dataTesting.load(100)
    i = 21