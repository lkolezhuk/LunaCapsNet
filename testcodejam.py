import numpy as np
from keras.utils.generic_utils import Progbar
import requests
import numpy as np
from io import StringIO
import base64
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from sklearn.model_selection import train_test_split
import conf
from utils import scannerMappingSim


from keras.utils import to_categorical
import matplotlib.pyplot as plt

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def stringifynp(x):
    output = StringIO.StringIO()
    np.save(output, x)
    output.seek(0)
    return base64.b64encode(output.read())


def submit(name, secret, predictions):
    payload = {
        'teamname': name,
        'secret': secret,
        'predictions': stringifynp(predictions)
    }
    r = requests.post('https://ec2-52-213-214-53.eu-west-1.compute.amazonaws.com/submit', json=payload, verify=False,
                      allow_redirects=True)
    return r.text


class code_jam_data(object):
    def __init__(self, root_path):
        self.root_path = root_path

    def load_subset(self, subset_index):
        return np.load("{}/{}.npz".format(self.root_path, subset_index))[:2]

    def load_score_images(self):
        return np.load("{}/8_images.npz".format(self.root_path))["images"]

    def load_final_images(self):
        filepath = "{}/9_images.npz".format(self.root_path)
        if not os.path.isfile(filepath):
            print
            "No final images"
            return None
        return np.load(filepath)["images"]

    def load_all_data(self):

        return self.load_subsets([9])

    def load_subsets(self, subsets):
        """
        Loads specified subsets of the data for the code jam.
        Returns tuple: ( images, labels, subset membership number )
        You can use the subset membership number to select the data from particular subset:
        e.g. result[(indices == 4).flatten()]
        """
        result = None
        resultLabels = None
        indices = None
        n_of_subsets = len(subsets)
        p = Progbar(n_of_subsets)
        p.update(0)
        for index, subsetIndex in enumerate(subsets):
            data = np.load("{}/{}.npz".format(self.root_path, subsetIndex))
            if result is None:
                result = data['images']
            else:
                result = np.vstack([result, data['images']])

            if resultLabels is None:
                resultLabels = data['labels']
            else:
                resultLabels = np.vstack([resultLabels, data['labels']])

            tmp = np.ones(data['labels'].shape) * subsetIndex
            if indices is None:
                indices = tmp
            else:
                indices = np.vstack([indices, tmp])
            p.update(index + 1)
        return (result, resultLabels, indices)


def select_subsets(data, indices, trainingSubsetIndex):
    selection = (indices != trainingSubsetIndex).flatten()
    negSelection = np.logical_not(selection)
    return (data[selection], data[negSelection])
from utils import sharpen
def dataf_mergeviews(data):
    x = []
    y = []
    for i in range(max(len(data[0]), len(data[1]))):
        images = data[0][i]
        label = data[1][i]
        for image in images:
            image = sharpen(image)
            # image = scannerMappingSim(image)
            # image = (image + 1000)/3000
            image = (image-image.mean())/image.std()

            # plt.imshow(image, cmap=plt.cm.gray)
            x.append(image)
            y.append(label)

    x = np.asarray(x)
    x = x.reshape(-1, conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1).astype('float32')

    y = np.asarray(y)
    y = to_categorical(y, conf.NUM_CLASSES)
    np.save('luna-images-merged9distorted2.npy', x)
    #np.save('luna-labels-merged9distorted1.npy', y)
    return(x, y)

def dataf_singleview(data, view):
    x = []
    y = []
    for i in range(max(len(data[0]), len(data[1]))):
        image = data[0][i][view]
        image = (image - image.mean()) / image.std()
        label = data[1][i]
        x.append(image)
        y.append(label)
    x = np.asarray(x)
    x = x.reshape(-1, conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1).astype('float32')

    y = np.asarray(y)
    y = to_categorical(y, conf.NUM_CLASSES)
    np.save('luna-images-single9-{0}.npy'.format(str(view)), x)
    np.save('luna-labels-single9.npy', y)
    return (x, y)

if __name__ == "__main__":
    d = code_jam_data("E:\LK\Data\LUNA16\candidates")
    data = d.load_all_data()
    x1, y1 = dataf_singleview(data, 0)
    x2, y2 = dataf_singleview(data, 1)
    x3, y3 = dataf_singleview(data, 2)

    import utils
    from PIL import Image
    images = []
    for i in range(0,5):
        images.append(np.array(x1[i][:, :, 0]))
        images.append(np.array(x2[i][:, :, 0]))
        images.append(np.array(x3[i][:, :, 0]))


    images = np.concatenate(images)
    plt.imshow(images, cmap='gray')
    plt.show()


    # plt.imshow(img, cmap='gray')
    # plt.show()
    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / images.max() * (images - images.min())).astype(np.uint8)
    Image.fromarray(rescaled).save('C:/Users/leko/Desktop/ct_views/' + 'patch-views.png')
    #
    #

    # for sample in data[0:5]:
    #     for ent in sample:
    #         images = []
    #         images.append(ent[0])
    #         images.append(ent[1])
    #         images.append(ent[2])
    #
    #         for i in range(len(images)):
    #             plt.imshow(images[i], cmap='gray')
    #             plt.show()
    #
    # x, y = dataf_mergeviews(data)


    # (x_train, x_val, y_train, y_val) = train_test_split(x, y, train_size=0.8)

    # print("Got {0} images and {1} labels".format(str(len(x)), str(len(y))))



    i = 1