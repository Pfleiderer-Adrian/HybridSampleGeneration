import distutils.dir_util
import os
import shutil
from datetime import datetime
import tensorflow as tf
from pandas import read_csv
from tensorflow.python.client import device_lib
from models.variationalAutoEncoder_ResNet import ResNetVAE, generate_samples_resnet
import configuration
from models.Detection_Model import Detection_Model
from utils.helpers import create_folder
from utils.prepare_dataset import copy_samples_in_one_folder, copy_labels, convert_dir_to_nii, \
    remove_samples_with_no_seg, \
    remove_control_samples, copy_control_samples, seperate_testdata, create_label_csv_from_labeldir, convert_gz_to_nii, \
    scale_labels_to_img
from utils.prepare_samples import match_and_fusion_anomalies, crop_and_center_anomaly
from utils.convnext_workaround import LayerScale, StochasticDepth

print(os.getcwd())
os.chdir(os.path.abspath("."))

# choose & test GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", device_lib.list_local_devices())
tf.test.gpu_device_name()

# check working dirs. Important for src backup
print("Home:"+os.path.abspath("."))
print("PATHS:"+os.environ['PATH'])
print("Lib:"+os.environ['LD_LIBRARY_PATH'])

# set or create config file
config_name = "gpu_server_config_3"
if not os.path.exists(os.path.join("configs", config_name+".json")):
    configuration.create_home_configuration(config_name)

# load config file
print("Load template configuration + generate session configuration")
mode = 0o777
config = configuration.load_config_file(config_name)
date = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
# config = configuration.create_new_config_from_template("config_" + str(date), config)
config.print_config()
config.write_config_file()

# global settings
anomaly_size = (96, 96, 16, 1)
modality = "t2w"
# modality = "hbv"
# anomaly_size = (36, 36, 16, 1)


# setup training environment
def step00_setup():
    print("STEP00: SETUP")
    config.print_config()

    # Create tmp folders for the data preprocessor
    print("Create folder structure")
    if not create_folder(config.path_to_output_folder, False):
        print("Output folder exists!")

    # tmp folders for internal processing
    config.tmp_dataset_path = os.path.join(config.path_to_output_folder, "train_set")
    create_folder(config.tmp_dataset_path, True)

    config.tmp_label_path = os.path.join(config.path_to_output_folder, "train_seg")
    create_folder(config.tmp_label_path, True)

    print("Copy images from source")
    copy_samples_in_one_folder(config.path_to_dataset, config.tmp_dataset_path, modalities=modality)
    print("Convert images from mha to nii datatype")
    convert_dir_to_nii(config.tmp_dataset_path, True)
    print("Copy labels from source")
    copy_labels(config.path_to_labels, config.tmp_label_path)
    print("Convert segmentations from .nii.gz to nii datatype")
    convert_gz_to_nii(config.tmp_label_path)
    print("Remove images without label file")
    remove_samples_with_no_seg(config.tmp_dataset_path, config.tmp_label_path)
    if modality != "t2w":
        print("Scale labels to image size")
        scale_labels_to_img(config.tmp_dataset_path, config.tmp_label_path)

    # separate test set from data source
    config.test_set = os.path.join(config.path_to_output_folder, "test_set")
    create_folder(config.test_set, True)
    config.test_seg = os.path.join(config.path_to_output_folder, "test_seg")
    create_folder(config.test_seg, True)
    # 33 positive, 192 negative test samples
    seperate_testdata(config.tmp_dataset_path, config.tmp_label_path, config.test_set, config.test_seg, [33, 192])

    config.write_config_file()


# setup cvae training session
def step11_setup_syn_model_training():
    print("STEP1.1: PREPARE SYN MODEL TRAINING")

    # create folder structure
    print("Create folder structure")
    if not os.path.exists(config.path_to_output_folder):
        raise Exception("Run folder not found! RUN STEP00 first.")

    config.syn_data_generation_path = os.path.join(config.path_to_output_folder, "Synthetic_Model_Training")
    if not create_folder(config.syn_data_generation_path, False):
        print("Synthetic_Model_Training-Folder already exists!")
        # raise Exception("Output folder exists!")

    tmp_set = os.path.join(config.syn_data_generation_path, "tmp_set")
    create_folder(tmp_set, True)

    tmp_seg = os.path.join(config.syn_data_generation_path, "tmp_seg")
    create_folder(tmp_seg, True)

    config.syn_train_data = os.path.join(config.syn_data_generation_path, "train_data")
    create_folder(config.syn_train_data, True)

    config.control_samples_path = os.path.join(config.syn_data_generation_path, "control_samples")
    create_folder(config.control_samples_path, True)

    config.syn_results = os.path.join(config.syn_data_generation_path, "results")
    create_folder(config.syn_results, False)

    print("Prepare Samples")
    distutils.dir_util.copy_tree(config.tmp_dataset_path, tmp_set)
    distutils.dir_util.copy_tree(config.tmp_label_path, tmp_seg)
    copy_control_samples(config.tmp_dataset_path, config.tmp_label_path, config.control_samples_path)
    # remove all negative samples from cvae training
    remove_control_samples(tmp_set, tmp_seg)
    # extract anomalys from positive samples for cvae training
    crop_and_center_anomaly(tmp_seg, tmp_set, config.syn_train_data, config, anomaly_size)
    shutil.rmtree(tmp_set)
    shutil.rmtree(tmp_seg)

    config.write_config_file()


# train cvae model
def step12_train_syn_gen_model():
    print("STEP1.2: TRAIN SYN MODEL")

    # create train folder structure
    print("Create folder structure")
    config.path_to_syn_run_folder = os.path.join(config.syn_results, date)
    create_folder(config.path_to_syn_run_folder, False)

    if config.syn_train_data == "":
        print("ERROR: run folder not found! RUN STEP1.1 first.")
        return

    print("Run model training")
    vae_resnet = ResNetVAE(input_shape=anomaly_size)
    vae_resnet.train_cvac(config.syn_train_data, config.path_to_syn_run_folder)

    config.write_config_file()


# generate new anomaly variants
def step13_generate_syn_from_anomalies():
    print("STEP1.3: GENERATE VARIANTS from real anomalies")

    config.generated_samples_path = os.path.join(config.path_to_syn_run_folder,
                                                 "generated_training_samples_from_anomalies")
    create_folder(config.generated_samples_path, False)

    generate_samples_resnet(os.path.join(config.path_to_syn_run_folder, "weights.tf"), config.syn_train_data,
                     config.generated_samples_path, anomaly_size, set_window_zero=True)

    config.write_config_file()


# fusion new anomaly into control samples
def step14_fusion_from_anomalies():
    config.fusioned_syn_samples_path = os.path.join(config.path_to_syn_run_folder, "fusioned_samples_from_anomalies")
    create_folder(config.fusioned_syn_samples_path, True)

    config.fusioned_syn_labels_path = os.path.join(config.path_to_syn_run_folder, "fusioned_labels_from_anomalies")
    create_folder(config.fusioned_syn_labels_path, True)

    # start fusion process
    match_and_fusion_anomalies(config.control_samples_path, config.generated_samples_path, config,
                               config.fusioned_syn_samples_path,
                               config.fusioned_syn_labels_path, 1)

    config.write_config_file()


# prepare detection model training
def step21_prepare_detection_model_training(fusioned_samples_path=None, fusioned_labels_path=None, overlap_control=True):
    print("STEP2.1: PREPARE DETECTION MODEL TRAINING")

    # create folder structure
    print("Create folder structure")
    if not os.path.exists(config.path_to_output_folder):
        raise Exception("Run folder not found! RUN STEP00 first.")

    config.detection_training_path = os.path.join(config.path_to_output_folder, "Detection_Model_Training")
    if not create_folder(config.detection_training_path, False):
        print("Output folder exists!")

    config.detection_results = os.path.join(config.detection_training_path, "results")
    create_folder(config.detection_results)

    config.path_to_detection_run_folder = os.path.join(config.detection_results, date)
    create_folder(config.path_to_detection_run_folder, False)

    # backup src files
    project_backup = os.path.abspath(os.path.abspath(config.path_to_detection_run_folder + "/src"))
    create_folder(project_backup)
    shutil.copytree(os.path.abspath("."), project_backup,
                    ignore=shutil.ignore_patterns("venv"), dirs_exist_ok=True)

    config.detection_img_train_data = os.path.join(config.path_to_detection_run_folder, "train_img_data")
    create_folder(config.syn_train_data, True)

    config.detection_seg_train_data = os.path.join(config.path_to_detection_run_folder, "train_seg_data")
    create_folder(config.syn_train_data, True)

    # copy original samples without test data to the training set
    distutils.dir_util.copy_tree(config.tmp_dataset_path, config.detection_img_train_data)
    distutils.dir_util.copy_tree(config.tmp_label_path, config.detection_seg_train_data)

    # copy generated samples to the training set
    # else train only with real samples
    if (fusioned_samples_path is not None) and (fusioned_labels_path is not None):
        distutils.dir_util.copy_tree(fusioned_samples_path, config.detection_img_train_data)
        distutils.dir_util.copy_tree(fusioned_labels_path, config.detection_seg_train_data)
        if overlap_control is False:
            data = read_csv(config.path_to_matching_infos)
            used_control = data['control'].tolist()
            for id in used_control:
                os.remove(os.path.join(config.detection_img_train_data, id))
                os.remove(os.path.join(config.detection_seg_train_data, id))

    # create label file for training
    config.csv_datapath = os.path.join(config.path_to_detection_run_folder, "labels_train.csv")
    create_label_csv_from_labeldir(config.detection_seg_train_data, config.csv_datapath, superclass=True)

    # create label file for evaluation
    config.csv_test_datapath = os.path.join(config.path_to_detection_run_folder, "labels_test.csv")
    create_label_csv_from_labeldir(config.test_seg, config.csv_test_datapath, superclass=True)

    config.write_config_file()


# train and evaluate detection model
def step22_initilize_detection_model(model_path=None):

    detection_model = Detection_Model(config)

    if model_path is not None:
        detection_model.model.load(
            os.path.join(model_path, "model.hdf5"),
            custom_objects={"LayerScale": LayerScale, "StochasticDepth": StochasticDepth})

    # start train & evaluation process
    detection_model.train()

    config.write_config_file()


# run pipeline
step00_setup()
step11_setup_syn_model_training()
step12_train_syn_gen_model()
step13_generate_syn_from_anomalies()
step14_fusion_from_anomalies()
# run with only real data
step21_prepare_detection_model_training()
step22_initilize_detection_model()
# run with real & synth. data
step21_prepare_detection_model_training(config.fusioned_syn_samples_path, config.fusioned_syn_labels_path)
step22_initilize_detection_model()