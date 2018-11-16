import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import os

from scipy import misc
from scipy import ndimage
import random
from sklearn.utils import shuffle




def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    fig.savefig('result/log.png')
    if show:
        plt.show()


# def combine_images(generated_images, height=None, width=None):
#     num = generated_images.shape[0]
#     if width is None and height is None:
#         width = int(math.sqrt(num))
#         height = int(math.ceil(float(num)/width))
#     elif width is not None and height is None:  # height not given
#         height = int(math.ceil(float(num)/width))
#     elif height is not None and width is None:  # width not given
#         width = int(math.ceil(float(num)/height))
#
#     shape = generated_images.shape
#     image = np.zeros((height*shape[0], width*shape[1]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index/width)
#         j = index % width
#         image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
#             img[:, :, 0]
#     return image
def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def hist_match(source, template):
    #histogram matching

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
def image_preprocessing(image):
    # # plt.imshow(image[0:, 0:, 0])
    # # plt.show()
    # # print("Mean: " + str(np.mean(image)) + " Std: " + str(np.std(image)))
    # image = tf.image.per_image_standardization(image)
    # sess = tf.Session()
    # with sess.as_default():
    #     image = np.array(image.eval())
    # # print("Mean: " + str(np.mean(image)) + " Std: " + str(np.std(image)))
    # # plt.imshow(image[0:, 0:, 0])
    # # plt.show()
    image = image/255.
    image = (image - image.mean()) / image.std()

    return image


from scipy.interpolate import interp1d

# maps original image intensities in HU scale to simulate different scanner
def scannerMappingSim(image):
    # plt.figure()
    # plt.imshow(image, cmap='gray')
    # plt.show()

    hu_points = [-3500,-750, -500, -50, 0, 150, 250, 1500, 3500]
    coef_points =[-0.08, -0.07, -0.064, -0.03, 0, 0.15, 0.2, 0.25, 0.26]
    interpolator = interp1d(hu_points, coef_points, 'quadratic')

    test_x = np.linspace(-750, 1500, 200)
    test_y = interpolator(test_x)

    # plt.figure()
    # plt.plot(test_x, test_y)
    # plt.show()


    newImage = [x * (1+2*interpolator(x)) for x in image]
    # plt.figure()
    # plt.imshow(newImage, cmap='gray')
    # plt.show()

    return np.array(newImage)
def frange(start, stop, step):
    i = start
    while i < stop:
         yield i
         i += step

def sharpen(image):
    # plt.figure()
    # plt.imshow(image, cmap='gray')
    # plt.show()

    sharpen_kernel = 30*np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    newImage = ndimage.convolve(image, sharpen_kernel, mode='reflect')
    # plt.figure()
    # plt.imshow(newImage, cmap='gray')
    # plt.show()
    return newImage
# def encode_chestx_labels(labels, positive_labels):
#     encoded_labels = []
#     for label in labels:
#         encoded_labels.append(encode_chestx_label(label, positive_labels))
#     return encoded_labels
# def encode_chestx_label(label, positive_labels, negative_labels):
#     labels = label.split('|')
#     for l in labels:
#         if l in positive_labels:
#             return 1
#         elif l in negative_labels:
#             return 0
# def encode_chestx_label_unique(label, positive_label, negative_label):
#         if label == positive_label:
#             return 1
#         elif label == negative_label:
#             return 0
#
# def image_locations_get_from_csv(csv_file_path, split=True, mode="TrainingValidation", modeReference=None):
#     if mode is "TrainingValidation" and modeReference is not None:
#         with open(csv_file_path) as csv_label_file:
#             with open(modeReference) as mode_ref_file:
#                 reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
#                 label_entities = list(reader)
#                 mode_entities = mode_ref_file.readlines()
#
#
#
#     elif mode is "Testing":
#         i = 1
#         # Do Something
#     with open(csv_file_path) as csv_label_file:
#         reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
#         label_entities = list(reader)
#         if split is True:
#             return image_locations_split_pos_neg(label_entities)
#         else:
#             return label_entities
#
# def image_locations_split_pos_neg(label_entities):
#     pos_entities = []
#     neg_entities = []
#     assert len(label_entities) > 0
#     for entity in label_entities:
#         if encode_chestx_label_unique(entity[1], "Mass", "No Finding") == 1:
#             pos_entities.append(entity)
#         elif encode_chestx_label_unique(entity[1], "Mass", "No Finding") == 0:
#             neg_entities.append(entity)
#     return (pos_entities, neg_entities)
#
# def get_data(data_size, pos_locations, neg_locations):
#     assert data_size > 1
#     assert len(pos_locations) > 0
#     assert len(neg_locations) > 0
#
#     filename = os.path.join(DATASET_PATH, "images/00000003_006.png")
#     hist_template = ndimage.imread(filename, True, 'L')
#
#     pos_indexes = random.sample(range(1, len(pos_locations)), int(data_size * CLASS_SPLIT))
#     neg_indexes = random.sample(range(1, len(neg_locations)), int(data_size * (1. - CLASS_SPLIT)))
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
#     for label_enry in pos_locations:
#         filename = os.path.join(DATASET_PATH, "images/" + label_enry[0])
#         try:
#             im = ndimage.imread(filename, True, 'L')
#             im = hist_match(im, hist_template)
#
#             im = misc.imresize(im, (IMAGE_X_DIM, IMAGE_Y_DIM), 'bilinear', 'L')
#             im = image_preprocessing(im)
#             im = im.reshape(IMAGE_X_DIM, IMAGE_Y_DIM, 1)
#
#
#             images.append(im)
#             labels.append(1)
#
#         except (FileNotFoundError, OSError):
#             print('Missing {0}'.format(str(filename)))
#         except ValueError as e:
#             print('Value error: ' + str(e))
#
#     for label_enry in neg_locations:
#         filename = os.path.join(DATASET_PATH, "images/" + label_enry[0])
#         try:
#             im = ndimage.imread(filename, True, 'L')
#             im = hist_match(im, hist_template)
#
#             im = misc.imresize(im, (IMAGE_X_DIM, IMAGE_Y_DIM), 'bilinear', 'L')
#             im = image_preprocessing(im)
#             im = im.reshape(IMAGE_X_DIM, IMAGE_Y_DIM, 1)
#
#
#             images.append(im)
#             labels.append(0)
#
#         except (FileNotFoundError, OSError):
#             print('Missing {0}'.format(str(filename)))
#         except ValueError as e:
#             print('Value error: ' + str(e))
#     # locations = pos_locations + neg_locations
#
#     images, labels = shuffle(images, labels)
#
#     x = np.asarray(images)
#     x = x.reshape(-1, IMAGE_X_DIM, IMAGE_Y_DIM, 1).astype('float32')
#
#     y = np.asarray(labels)
#
#     return x, y
# def get_data_validation(data_size, locations):
#     assert data_size > 1
#     assert len(locations) > 0
#
#     filename = os.path.join(DATASET_PATH, "images/00000003_006.png")
#     hist_template = ndimage.imread(filename, True, 'L')
#
#     indexes = random.sample(range(1, len(locations)), int(data_size))
#
#     locations_selected = []
#     for ind in indexes:
#         locations_selected.append(locations[ind])
#
#     locations = locations_selected
#
#     images = []
#     labels = []
#
#     for label_enry in locations:
#         filename = os.path.join(DATASET_PATH, "images/" + label_enry[0])
#         try:
#             im = ndimage.imread(filename, True, 'L')
#             im = hist_match(im, hist_template)
#
#             im = misc.imresize(im, (IMAGE_X_DIM, IMAGE_Y_DIM), 'bilinear', 'L')
#             im = image_preprocessing(im)
#             im = im.reshape(IMAGE_X_DIM, IMAGE_Y_DIM, 1)
#
#             images.append(im)
#             labels.append(1)
#
#         except (FileNotFoundError, OSError):
#             print('Missing {0}'.format(str(filename)))
#         except ValueError as e:
#             print('Value error: ' + str(e))
#
#     images, labels = shuffle(images, labels)
#
#     x = np.asarray(images)
#     x = x.reshape(-1, IMAGE_X_DIM, IMAGE_Y_DIM, 1).astype('float32')
#
#     y = np.asarray(labels)
#
#     return x, y


# def split_pos_neg_to_folder(folder_path):
#
#     missing_files_count = 0
#     with open(os.path.join(DATASET_PATH, 'Data_Entry_2017.csv')) as csv_label_file:
#         reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
#         label_entities = list(reader)
#
#         for entry in label_entities[1:]:
#             filename = os.path.join(DATASET_PATH, "images/" + entry[0])
#             label = encode_chestx_label(entry[1], ["Mass", "Nodule"])
#
#             if (not (os.path.exists(folder_path + "pos/" + str(entry[0]))
#                      or os.path.exists(folder_path + "neg/" + str(entry[0])))
#                      and os.path.exists(filename)):
#                 try:
#                     im = scipy.ndimage.imread(filename, True, 'L')
#                     # im = scipy.misc.imresize(im, (50, 50), 'bilinear', 'L')
#
#                     if label == 1:
#                         scipy.misc.imsave(folder_path + "pos/" + str(entry[0]), im)
#                     else:
#                         scipy.misc.imsave(folder_path + "neg/" + str(entry[0]), im)
#
#                 except (FileNotFoundError, OSError):
#                     # print('Error reading file: ' + filename)
#                     missing_files_count = missing_files_count + 1
#                     # print('Missing {0} files'.format(str(missing_files_count)))
#                 except ValueError as e:
#                     print('Value error: ' + str(e))
#
#     print('Finished spliting dataset into folders. The result is in:' + str(folder_path))
#     if missing_files_count > 0:
#         print('{0} files were missing'.format(str(missing_files_count)))


if __name__=="__main__":
    #
    # (pos_entities, neg_entities) = image_locations_get_from_csv("E:/XRay/Data_Entry_2017.csv")
    # (images, labels) = get_data(100, pos_entities, neg_entities)
    # for im in images:
    #     im = im[0:,0:,0]
    #     mean = np.mean(im)
    #     std = np.std(im)
    #     print("mean: " + str(mean) + " std: " + str(std))
    #
    # # filename="E:/XRay/images/00000003_006.png"
    # # hist_template = ndimage.imread(filename, True, 'L')
    # #
    # # filename = "E:/XRay/images/00000004_000.png"
    # # hist_target = ndimage.imread(filename, True, 'L')
    # # matched = hist_match(hist_target , hist_template);
    # # plt.imshow(matched,cmap=plt.cm.gray)
    # # plt.show()
    # it = 0
    # for i in images:
    #     plt.imshow(i[0:,0:,0], cmap=plt.cm.gray)
    #     plt.title("Label: " + str(labels[it]))
    #     plt.show()
    #     it = it+1

    # entities = image_locations_get_from_csv("E:/XRay/Data_Entry_2017.csv", split=False)
    # (val_img, val_labels) = get_data_validation(1000, entities)
    #
    #
    #
    #
    #
    #
    # iii = 1
