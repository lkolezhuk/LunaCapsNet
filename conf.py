
IMAGE_X_DIM = 32
IMAGE_Y_DIM = 32

DATASET_SAMPLES = 430000
BATCH_SIZE = 32
CLASS_SPLIT = 0.5 #of positive class

STEPS_PER_EPOCH = 1000
VALIDATION_STEPS = 250



DATASET_PATH = "/home/leo/Dev/XRay/"
# DATASET_PATH = "E:/XRay/"
DATA_TRAINVAL_FILENAME = "train_val_list.txt"
DATA_TESTING_FILENAME = "test_list.txt"
DATA_LABELS_FILENAME = "Data_Entry_2017.csv"
DATA_LABELS_TRAINVAL_FILENAME = "Data_Entry_TrainVal.csv"
DATA_LABELS_TESTING_FILENAME = "Data_Entry_Testing.csv"

NUM_CLASSES = 2
POSITIVE_LABELS = ["Mass"]
# "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Pneumonia", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NEGATIVE_LABELS = ["No Finding"]