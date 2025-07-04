import os
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi import *
import pandas as pd
from aucmedi.data_processing.subfunctions import Crop, Standardize, Chromer, Padding, Clip, Resize
from aucmedi.evaluation import evaluate_fitting, evaluate_performance
from aucmedi.neural_network.loss_functions import categorical_focal_loss, binary_focal_loss
from aucmedi.sampling import sampling_split
from aucmedi.utils.class_weights import compute_class_weights
import tensorflow.python.keras.callbacks as cb
from utils.prepare_dataset import create_label_csv_from_labeldir
from utils.convnext_workaround import LayerScale, StochasticDepth


# global config
# t2w
input_shape = (260, 260, 32)
# hdv
# input_shape = (128, 128, 32)
batch_size = 12


class Detection_Model:

    def __init__(self, config):
        self.config = config
        mode = 0o777
        
        # Initialize input data reader
        interface = input_interface(interface="csv",
                                    path_imagedir=self.config.detection_img_train_data,
                                    path_data=self.config.csv_datapath,
                                    )
        (index_list, class_ohe, nclasses, self.class_names, self.image_format) = interface

        self.ds = sampling_split(index_list, class_ohe, sampling=[0.8, 0.2])

        # Compute class weights
        cw_loss, self.cw_fit = compute_class_weights(class_ohe)

        #  Initialize a 3D ConvNeXtSmall model with ImageNet weights
        self.model = NeuralNetwork(n_labels=2, channels=3, learning_rate=0.01,
                                   architecture="3D.ConvNeXtSmall", pretrained_weights=True,
                                   loss=categorical_focal_loss(cw_loss), input_shape=input_shape)

        # self.model.tf_lr_end = 1e-6

        # create sub-functions
        padding_sf = Padding(mode="constant", shape=input_shape)
        crop_sf = Crop(shape=input_shape, mode="center")
        clipping_sf = Clip(min=0, max=1000)
        normalize_sf = Standardize(mode='minmax')
        normalize_sf = Standardize(mode='grayscale', smooth=1e-06)
        chromer_sf = Chromer(target='rgb')
        self.subfunctions = [clipping_sf, normalize_sf, padding_sf, crop_sf, chromer_sf]

        # create augmentation-functions
        self.augmentation = BatchgeneratorsAugmentation(
            image_shape=input_shape,
            rotate=True,
            brightness=True,
            contrast=True,
            scale=True,
            gaussian_noise=True,
            gamma=False,
            elastic_transform=True,
        )

        self.resampling_shape = (0.5, 0.5, 5)

    def train(self):
        # Initialize training Data Generator
        train_gen = DataGenerator(samples=self.ds[0][0],
                                  path_imagedir=self.config.detection_img_train_data,
                                  labels=self.ds[0][1],
                                  image_format=self.image_format,
                                  grayscale=True,
                                  resize=None,
                                  standardize_mode=self.model.meta_standardize,
                                  subfunctions=self.subfunctions,
                                  loader=sitk_loader,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=0,
                                  resampling=self.resampling_shape,
                                  data_aug=self.augmentation)

        val_gen = DataGenerator(samples=self.ds[1][0],
                                path_imagedir=self.config.detection_img_train_data,
                                labels=self.ds[1][1],
                                image_format=self.image_format,
                                grayscale=True,
                                resize=None,
                                standardize_mode=self.model.meta_standardize,
                                subfunctions=self.subfunctions,
                                loader=sitk_loader,
                                shuffle=False,
                                batch_size=batch_size,
                                seed=0,
                                resampling=self.resampling_shape,
                                data_aug=self.augmentation)

        # Create a callback that saves the best model's weights
        model_path = os.path.join(self.config.path_to_detection_run_folder, "model.hdf5")
        cb_cp = cb.ModelCheckpoint(
            filepath=model_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            mode="min",
            monitor="val_loss"
        )

        # create callback-functions
        # Adjust learning rate when our model hits a plateau (reduce overfitting)
        cb_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=7,
                                     verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                                     min_lr=0.0000001)
        cb_es = cb.EarlyStopping(monitor="val_loss", patience=10)
        cb_cl = cb.CSVLogger(os.path.join(self.config.path_to_detection_run_folder, "training_log.csv"), separator=',',
                             append=True)
        callbacks = [cb_cp, cb_es, cb_cl, cb_lr]

        # Run model training with transfer learning
        history = self.model.train(train_gen, val_gen, epochs=100, transfer_learning=True,
                                   callbacks=callbacks)

        evaluate_fitting(history, self.config.path_to_detection_run_folder)

        self.eval_model(model_path, True)

    # evaluate trained model
    def eval_model(self, model_path, external_test_data=False):
        if external_test_data:
            interface = input_interface(interface="csv",
                                        path_imagedir=self.config.test_set,
                                        path_data=self.config.csv_test_datapath,
                                        )
            (index_list, class_ohe, nclasses, self.class_names, self.image_format) = interface

            self.ds = sampling_split(index_list, class_ohe, sampling=[1])

        # Initialize testing Data Generator
        test_gen = DataGenerator(samples=self.ds[0][0],
                                 path_imagedir=self.config.test_set,
                                 image_format=self.image_format,
                                 grayscale=True,
                                 resize=None,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 seed=0,
                                 resampling=self.resampling_shape,
                                 standardize_mode=self.model.meta_standardize,
                                 subfunctions=self.subfunctions,
                                 loader=sitk_loader,
                                 data_aug=self.augmentation)
        # Run model inference for unknown samples
        self.model.load(model_path, custom_objects={"LayerScale": LayerScale, "StochasticDepth": StochasticDepth})
        pred = self.model.predict(test_gen)
        pd.DataFrame(pred).to_csv(os.path.join(self.config.path_to_detection_run_folder, "test_result.csv"))
        evaluate_performance(pred, self.ds[0][1], out_path=self.config.path_to_detection_run_folder,
                             class_names=self.class_names)

    # evaluate hbv & t2w in one evaluation
    def eval_combined_models(self, csv_path, output_path):
        interface = input_interface(interface="csv",
                                    path_imagedir=self.config.test_set,
                                    path_data=self.config.csv_test_datapath,
                                    )
        (index_list, class_ohe, nclasses, self.class_names, self.image_format) = interface

        self.ds = sampling_split(index_list, class_ohe, sampling=[1])
        df = pd.read_csv(csv_path).to_numpy()
        print(df)
        evaluate_performance(df, self.ds[0][1], out_path=output_path,
                             class_names=self.class_names)
