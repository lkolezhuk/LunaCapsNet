"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers, losses, metrics, activations
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import utils
import tensorflow as tf
from PIL import Image
import cv2
import scipy
from scipy import misc
import csv
from sklearn.model_selection import train_test_split
from data_retriever import DataRetriever
import conf
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from sklearn.utils import shuffle


K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1')(x)
    dp1 = layers.Dropout(0.1)(conv1)

    # Layer 2: Primary Capsule layer
    primarycaps = PrimaryCap(dp1, dim_capsule=8, n_channels=32, kernel_size=5, strides=2, padding='valid')

    # Layer 3: Capsule layer
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: Conversion layer
    # outputs = layers.Reshape(target_shape=[-1, 64], name='primarycap_reshape')(conv5)
    out_caps = Length(name='capsnet')(digitcaps)

    # out = layers.Dense(2, activation='sigmoid')(out_caps)

    # Reconstruct the initial input from the encoded capsule representation
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y]) # select one capsule by using GT
    masked = Mask()(digitcaps)

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))

    decoder.add(layers.Dense(1024, activation='relu'))

    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # capsule representation noise model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

    # return K.mean(K.binary_crossentropy(y_pred, y_true), axis=1)


    #
    # w_neg = 1.
    # w_pos = 1.
    #
    #
    # L = - y_true * K.log(y_pred) - (1 - y_true) * K.log(y_pred)
    # # L = L - y_true * K.log(1-y_pred[1]) - (1 - y_true) * K.log(y_pred[0])
    #
    # # raise Exception(str(R.shape))
    # return L

def spread_loss(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1.0 - y_true) * y_pred, axis=-1)
    return K.maximum(0.0, neg - pos + 1)


def train(model, args):
    # output callback func
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=False, save_weights_only=False, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})



    # x = np.load('sNORB_img_train_azimuths30_32_34_0_2_4.npy')
    # y = np.load('sNORB_lbl_train_azimuths30_32_34_0_2_4.npy')
    x = np.load('luna-images-merged08.npy')
    y = np.load('luna-labels-merged08.npy')

    x = np.array([resizeIMG(i) for i in np.array(x)])

    # pos_cnt = 0
    # for lab in y:
    #     if(np.argmax(lab) == 1):
    #         pos_cnt += 1
    #     # print(str(lab))
    # print("Positive: {0}, negative {1}".format(str(pos_cnt), str(len(y)-pos_cnt)))
    (train_images, val_images, train_labels, val_labels) = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=45)
    print("Using {0} images for training and {1} for validation.".format(str(len(train_images)), str(len(val_images))))

    gen_b = batch_generator(train_images, train_labels)
    gen_v = validation_generator(val_images, val_labels)
    model.fit_generator(generator=gen_b,
                        steps_per_epoch=conf.STEPS_PER_EPOCH,
                        epochs=args.epochs,
                        validation_data=gen_v,
                        validation_steps=conf.VALIDATION_STEPS,
                        callbacks=[log, tb, checkpoint, lr_decay])



    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=False)

    return model
def test(model, data, args):
    print('-' * 30 + 'Begin: test' + '-' * 30)

    x_test, y_test = data
    print('Testing on {0} images'.format(len(y_test)))
    print (np.argmax(y_test, axis=1))
    y_pred, x_recon = model.predict(x_test)


    true_lab = []
    pred_lab = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, len(y_pred)):
        t = np.argmax(y_test[i])
        p = np.argmax(y_pred[i])
        print("GT: " + str(t) + " Pred: " + str(p) + "--------- GT: " + str(y_test[i]) + " Pred: " + str(y_pred[i]))

        true_lab.append(t)
        pred_lab.append(p)

        if t == p == 1:
            TP += 1
        if p == 1 and t != p:
            FP += 1
        if t == p == 0:
            TN += 1
        if p == 0 and t != p:
            FN += 1
        print("GT: " + str(t) + " Pred: " + str(p) + "--------- GT: " + str(y_test[i]) + " Pred: " + str(y_pred[i]))

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    # conf_mat = confusion_matrix(y_test, y_pred)

    print("Accuracy {0}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))

    roc = roc_auc_score(true_lab, pred_lab)
    print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP, FP, TN, FN))
    print("Accuracy: {0}".format((TP + TN) / (TP + TN + FP + FN)))
    print("AUC ROC : " + str(roc))
    print("F1 score: {0}".format(2 * TP / (2 * TP + FP + FN)))
    print("Sensitivity(TPR): {0}, Specificity(TNR): {1}".format(TP / (TP + FN), TN / (FP + TN)))
    print("PPV(Precision): {0}".format(TP / (TP + FP)))
    print("NPV: {0}".format(TN / (TN + FN)))

    img = utils.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    Image.fromarray(rescaled).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()
    # images = np.concatenate(x_test[0:50,0:], x_recon[0:50,0:])
    # combined_image = utils.combine_images(images)
    # misc.imsave("result/reconstruction_result.png", combined_image)
def test_ensemble(model, args):
    print('-' * 30 + 'Begin: test' + '-' * 30)
    y_test = np.load('luna-labels-single9.npy')
    predictions = []
    for view in range(0, 3):
        print("Prediction for view {0}". format(view))
        x = np.load('luna-images-single9-{0}.npy'.format(view))
        y_pred, x_recon = model.predict(x)
        predictions.append(y_pred)

    y_pred = []
    for i in range(len(predictions[0])):
        neg_prob = (1.*predictions[0][i][0] + 1.*predictions[1][i][0] + 1.*predictions[2][i][0]) / 3.
        pos_prob = (1. * predictions[0][i][1] + 1. * predictions[1][i][1] + 1. * predictions[2][i][1]) / 3.
        print("{0}=={1}=={2}; {3}=={4}=={5}".format(predictions[0][i][0], predictions[1][i][0], predictions[2][i][0], predictions[0][i][1], predictions[1][i][1], predictions[2][i][1]))

        avg_pred = [neg_prob, pos_prob]
        y_pred.append(avg_pred)


    true_lab = []
    pred_lab = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with open('predictions.csv','wb') as csv_file:
        for i in range(0, len(y_pred)):
            t = np.argmax(y_test[i])
            p = np.argmax(y_pred[i])
            writer = csv.writer(csv_file, delimiter=' ')
            writer.writerow([str(y_pred[i])])

            true_lab.append(t)
            pred_lab.append(p)

            if t == p == 1:
                TP += 1
            if p == 1 and t != p:
                FP += 1
            if t == p == 0:
                TN += 1
            if p == 0 and t != p:
                FN += 1
            # print("GT: " + str(t) + " Pred: " + str(p) + "--------- GT: " + str(y_test[i]) + " Pred: " + str(y_pred[i]))

    from sklearn.metrics import roc_auc_score

    roc = roc_auc_score(true_lab, pred_lab)
    print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP, FP, TN, FN))
    print("Accuracy: {0}".format((TP + TN) / (TP + TN + FP + FN)))
    print("AUC ROC : " + str(roc))
    print("F1 score: {0}".format(2 * TP / (2 * TP + FP + FN)))
    print("Sensitivity(TPR): {0}, Specificity(TNR): {1}".format(TP / (TP + FN), TN / (FP + TN)))
    print("PPV(Precision): {0}".format(TP / (TP + FP)))
    print("NPV: {0}".format(TN / (TN + FN)))

    # img = utils.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    # image = img*255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    # print()
    # print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    # print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()
    # images = np.concatenate(x_test[0:50,0:], x_recon[0:50,0:])
    # combined_image = utils.combine_images(images)
    # misc.imsave("result/reconstruction_result.png", combined_image)
def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, conf.NUM_CLASSES, 16])
    x_recons = []
    for dim in range(16):
        # for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        for r in utils.frange(-0.25, 0.25, 0.05):
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = utils.combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)

def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)
def load_chestxray():
    image_list = []
    filenames_list = []
    label_list = []
    dataset_path = 'D:/XRAYchest/'

    with open(os.path.join(dataset_path, 'Data_Entry_2017.csv')) as csv_label_file:
        reader = csv.reader(csv_label_file, delimiter=',', quotechar='|')
        label_entities = list(reader)

        for entry in label_entities[1:100]:
            filename = os.path.join(dataset_path, "images/" + entry[0])
            label = entry[1]
            try:
                im = scipy.ndimage.imread(filename, True, 'L')
                im = scipy.misc.imresize(im, (50, 50), 'bilinear', 'L')

                # plt.imshow(np.float32(im),cmap='gray')
                # plt.show()

                image_list.append(np.matrix(im))
                label_list.append(label)
                filenames_list.append(filename)
            except (FileNotFoundError, OSError) :
                print('Could not find file listed in the label reference  ' + filename)

    x = np.asarray(image_list)
    x = x.reshape(-1, 50, 50, 1).astype('int16') / 255
    print('x shape' + str(x.shape))
    y = utils.encode_chestx_labels(label_list, ["Mass"])
    y = np.asarray(y)
    y = to_categorical(y.astype('int'))
    print('y shape' + str(y.shape))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


    return (x_train, y_train), (x_test, y_test)


def medianFilter(img):
    img[:64,:64,0] = cv2.medianBlur(img[:64,:64,0],5)
    return img

def resizeIMG(image):
    target_size = (32, 32)
    xs, ys = target_size
    temp = misc.imresize(image[:64, :64, 0], target_size, 'bilinear')
    image[:xs,:ys,0] = (temp - temp.mean()) / temp.std()

    return image[:xs,:ys]

def center_crop(img, center_crop_size, **kwargs):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 1

    x,y = img.shape[0]//2, img.shape[1]//2
    halfw, halfh = center_crop_size//2, center_crop_size//2
    return normalize(img[x-halfw:x+halfw, y-halfh:y+halfh, :])

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return normalize(img[y:(y+dy), x:(x+dx), :])

def normalize(img):
    temp = img[:, :, 0]
    img[:, :, 0] = (temp - temp.mean()) / temp.std()
    return img
def crop_generator(batches, crop_length):
    '''
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator
    '''
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 1))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))

        yield (batch_crops, batch_y)
def batch_generator(images, labels):
    # (pos_entities, neg_entities) = utils.image_locations_get_from_csv(os.path.join(utils.DATASET_PATH, "Data_Entry_2017.csv"))
    # (images, labels) = utils.get_data(utils.DATASET_SAMPLES, pos_entities, neg_entities)
    # # (images, labels),  (x_test, y_test) = load_mnist()
    # labels = to_categorical(labels, 10)
    generator = ImageDataGenerator(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0.,
                     width_shift_range=0.0,
                     height_shift_range=0.0,
                     shear_range=0.,
                     zoom_range=0.)


    # generator.fit(images)

    gen = generator.flow(images, labels, batch_size=conf.BATCH_SIZE)
    while 1:
        x_batch, y_batch = gen.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
    # batch_cropped = crop_generator(gen, 32)
    #
    # while 1:
    #     x_batch, y_batch = batch_cropped.next()
    #     yield ([x_batch, y_batch], [y_batch, x_batch])

def validation_generator(images, labels):
    # (pos_entities, neg_entities) = utils.image_locations_get_from_csv(os.path.join(utils.DATASET_PATH, "Data_Entry_2017.csv"))
    # (images, labels) = utils.get_data(size, pos_entities, neg_entities)
    # # (images, labels), (x_test, y_test) = load_mnist()
    # labels = to_categorical(labels, 10)

    generator = ImageDataGenerator(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     zoom_range=0.)


    # generator.fit(images)

    gen = generator.flow(images, labels, batch_size=conf.BATCH_SIZE)

    while 1:
        x_batch, y_batch = gen.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
    # batch_cropped = crop_generator(gen, 32)
    #
    # while 1:
    #     x_batch, y_batch = batch_cropped.next()
    #     yield ([x_batch, y_batch], [y_batch, x_batch])



if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="CapsNetLK")
    parser.add_argument('--epochs', default=10, type=int, help="Epochs")
    parser.add_argument('--batch_size', default=conf.BATCH_SIZE, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--lr_decay', default=0.8, type=float, help="Learning rate decay")
    parser.add_argument('--lam_recon', default=0.512, type=float, help="Reconstruction weight") #0.392 default
    parser.add_argument('-r', '--routings', default=3, type=int, help="Routing iterations")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true')
    parser.add_argument('-w', '--weights', default=None)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # (x_train, y_train), (x_test, y_test) = load_chestxray()

    # # load data
    # #(x_train, y_train), (x_test, y_test) = load_mnist()


    model, eval_model, manipulate_model = CapsNet(input_shape=(conf.IMAGE_X_DIM, conf.IMAGE_Y_DIM, 1),
                                                  n_class=conf.NUM_CLASSES,
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:

        train(model=model, args=args)


    else:  # as long as weights are given, wi
        # ll run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')


        # dataset = DataRetriever(testing=True)
        # dataset.load(50)
        # (images, labels) = dataset.getTest()
        x = np.load('luna-images-merged9.npy')
        y = np.load('luna-labels-merged9.npy')

        #
        # x = np.load('sNORB_img_test_azimuths6_28.npy')
        # y = np.load('sNORB_lbl_test_azimuths6_28.npy')

        #
        # x = np.load('sNORB_img_test_azimuths30_32_34_0_2_4.npy')
        # y = np.load('sNORB_lbl_test_azimuths30_32_34_0_2_4.npy')

        x = np.array([resizeIMG(i) for i in x])
        # x = np.array([center_crop(i, 32) for i in x])


        x,y = shuffle(x, y)
        manipulate_latent(manipulate_model, (x, y), args)
        # test(model=eval_model, data=(x, y), args=args)
        # test_ensemble(model=eval_model, args=args)