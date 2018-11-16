import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from scipy import misc
import conf
import glob
import progressbar
import ntpath



class smallNorb:
    def __init__(self, mode='test'):
        self.mode = mode
        self.labels = {'animal':0, 'human':1, 'airplane':2, 'truck':3, 'car':4}
        self.images_folder = "D:/smallNORB/small_norb-master/smallnorb_export3"

        if mode == 'test' :
            self.images_folder = os.path.join(self.images_folder, 'test')
        else:
            self.images_folder = os.path.join(self.images_folder, 'train')

    def filterAzimuthsAndElevations(self, files, azimuths=[30, 32, 34, 0, 20, 40], elevations=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        azimuths = ["{:02d}".format(f) for f in azimuths]
        elevations = ["{:02d}".format(f) for f in elevations]

        def filterEachAE(str):
            data = ntpath.basename(str)
            data = data.split('_')
            if data[3] in azimuths:
                if data[4] in elevations:
                    return True
            return False
        return [f for f in files if filterEachAE(f)]

    def load(self):
        images = []
        labels = []
        bar = progressbar.ProgressBar()
        files = glob.glob(os.path.join(self.images_folder, "*.jpg"))

        files_subset = self.filterAzimuthsAndElevations(files)
                                                        # , azimuths = list(range(6, 29)), elevations = list(range(0, 9)))
        for file in bar(files_subset):
            file_info = os.path.basename(file).split('_')

            im = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            im = misc.imresize(im, (conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM), 'bilinear')

            # th, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            im = im / 255.
            im = (im - im.mean()) / im.std()

            images.append(im)
            labels.append(self.labels[file_info[1]])
            # plt.figure()
            # plt.imshow(im, cmap='gray')
            # plt.show()


        labels = np.array(labels)
        labels = to_categorical(labels)

        images = np.asarray(images)
        images = images.reshape(-1, conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1)


        np.save('sNORB_img_{0}_azimuths30_32_34_0_2_4.npy'.format(self.mode), images)
        np.save('sNORB_lbl_{0}_azimuths30_32_34_0_2_4.npy'.format(self.mode), labels)

        return images, labels

if __name__ == "__main__":
    reader = smallNorb(mode='test')
    (images, labels) = reader.load()

    # images = np.load('sNORB_img_test_bin1.npy')
    # images2 = np.load('sNORB_img_test_bin2.npy')
    # labels = np.load('sNORB_lbl_test_bin1.npy')
    # labels2 = np.load('sNORB_lbl_test_bin2.npy')
    #
    # img = np.append(images, images2, axis=0)
    # lbl = np.append(labels, labels2, axis=0)
    # #
    # np.save('sNORB_img_test_azimuths_bin.npy', img)
    # np.save('sNORB_lbl_test_azimuths_bin.npy', lbl)
    # # #
    # i = np.load('sNORB_img_test_azimuths.npy')
    # l = np.load('sNORB_lbl_test_azimuths.npy')

    # for ii in range(1100):
    #     im = i[ii][:64,:64,0]
    #
    #     im = im*255
    #     im = im.astype(np.uint8)
    #     # th,im = cv2.threshold(im, 75, 0, cv2.THRESH_BINARY)
    #     # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    #     im = cv2.medianBlur(im, 3)
    #     plt.figure()
    #     plt.imshow(im, cmap='gray')
    #     plt.show()
    # pass