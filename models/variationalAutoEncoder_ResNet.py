import glob
import os
import shutil
import numpy as np
import tensorflow as tf
import nibabel as nib
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import SGD, Adam
from tensorboard import program
from tensorflow import keras
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import backend as K
import datetime
from utils import visualizer

# global config
learning_rate_1 = 0.00001
learning_rate_2 = 0.001
momentum = 0.9
kernel_size = 3
batch_size = 32
epochs = 2000


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, depth, kernel_size=(3, 3, 3), stride=(1, 1, 1), **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)

        self.residual_block = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same', name="ResBlock"+str(depth)+"_Conv1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2, name="ResBlock"+str(depth)+"_LRelu1"),
            tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same', name="ResBlock"+str(depth)+"_Conv2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2, name="ResBlock"+str(depth)+"_LRelu2"),
        ])

    def call(self, x, **kwargs):
        return x + self.residual_block(x)


class ResNetEncoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 bUseMultiResSkips=True, **kwargs):

        super(ResNetEncoder, self).__init__(**kwargs)

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters_1, i)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv3D(filters=n_filters_2, kernel_size=(2, 2, 2),
                                           strides=(2, 2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv3D(filters=self.max_filters, kernel_size=(ks, ks, ks),
                                               strides=(ks, ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv3D(filters=z_dim, kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1), padding='same')

    def call(self, x, **kwargs):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 output_channels=3,
                 bUseMultiResSkips=True, **kwargs):

        super(ResNetDecoder, self).__init__(**kwargs)

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=self.max_filters, kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1), padding='same', name="Decoder"+"_InputConv"),
            tf.keras.layers.BatchNormalization(name="Decoder"+"_InputBNorm"),
            tf.keras.layers.LeakyReLU(alpha=0.2, name="Decoder"+"_InputLRelu"),
        ])

        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters, i)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv3DTranspose(filters=n_filters, kernel_size=(2, 2, 2),
                                                    strides=(2, 2, 2), padding='same', name="Decoder"+"_ConvT"+str(i)),
                    tf.keras.layers.BatchNormalization(name="Decoder"+"_BNorm"+str(i)),
                    tf.keras.layers.LeakyReLU(alpha=0.2, name="Decoder"+"_LRelu"+str(i)),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv3DTranspose(filters=n_filters, kernel_size=(ks, ks, ks),
                                                        strides=(ks, ks, ks), padding='same', name="Decoder"+"_ResSkip_ConvT"+str(i)),
                        tf.keras.layers.BatchNormalization(name="Decoder"+"_ResSkip_BNorm"+str(i)),
                        tf.keras.layers.LeakyReLU(alpha=0.2, name="Decoder"+"_ResSkip_LRelu"+str(i)),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(3, 3, 3),
                                                  strides=(1, 1, 1), padding='same', name="Decoder"+"_Output_ConvT")

    def call(self, z, **kwargs):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)
        return z


class ResNetVAE(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(256, 256, 16, 1),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=250,
                 bottleneck_dim=250,
                 bUseMultiResSkips=True, **kwargs):
        super(ResNetVAE, self).__init__(**kwargs)
        self.anomaly_shape = input_shape
        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[3]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.img_latent_dim_z = input_shape[2] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=output_channels, bUseMultiResSkips=bUseMultiResSkips)

        self.fc21 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc22 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc3 = tf.keras.layers.Dense(self.img_latent_dim_z * self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = tf.keras.backend.reshape(h1, shape=(-1, self.img_latent_dim_z * self.img_latent_dim * self.img_latent_dim * self.z_dim))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = tf.keras.backend.exp(0.5*logvar)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps*std

    def decode(self, z):
        z = self.fc3(z)
        z = tf.keras.backend.reshape(z, shape=(-1, self.img_latent_dim, self.img_latent_dim, self.img_latent_dim_z, self.z_dim))
        h3 = self.decoder(z)
        return tf.keras.backend.sigmoid(h3)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)

        reconstructed_flatten = tf.keras.layers.Flatten()(reconstructed)
        x_flatten = tf.keras.layers.Flatten()(x)

        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)

        # testing different loss functions and combinations
        #reconstruction_loss = mse(x_flatten, reconstructed_flatten)
        #reconstruction_loss = tensorflow.image.ssim_multiscale(x, reconstructed, k2=0.3)
        reconstruction_loss = keras.metrics.binary_crossentropy(x_flatten, reconstructed_flatten)
        reconstruction_loss *= self.anomaly_shape[0] * self.anomaly_shape[1] * self.anomaly_shape[2]
        #kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        vae_loss = K.mean(100 * reconstruction_loss + kl_loss)
        #vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.add_loss(vae_loss)

        return reconstructed


    def show_details(self):
        self.encoder.build(input_shape=self.input_shape)
        self.encoder.summary()
        self.decoder.build(input_shape=(self.img_latent_dim_z * self.img_latent_dim * self.img_latent_dim * self.z_dim))
        self.decoder.summary()

    def train_cvac(self, img_path, output_dir_path):
        sgd = SGD(lr=learning_rate_1, momentum=momentum, nesterov=True)
        adam = Adam(lr=learning_rate_1)
        self.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=learning_rate_1))

        train_ds, test_ds = load_dataset(img_path)

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=50,
            min_lr=0.00001)

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
        # lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        self.fit(train_ds,
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[reduce_lr, create_tensorboard(output_dir_path), early_stopping],
                 validation_split=0.20,
                 validation_data=(train_ds, None)
                 )

        generated_images, input_images = self.test_model(test_ds, 10)

        plot = visualizer.plot_slices(generated_images, input_images, slice_axis=3)

        plot.savefig(os.path.join(os.path.abspath(output_dir_path), "res_plot.png"))
        self.save_weights(os.path.join(os.path.abspath(output_dir_path), "weights.tf"), save_format="tf")

        project_backup = os.path.abspath(os.path.abspath(output_dir_path + "/src"))
        shutil.copytree(os.path.abspath("."), project_backup,
                        ignore=shutil.ignore_patterns("venv"), dirs_exist_ok=True)


    def test_model(self, data, num_samples=10):
        # Select a random batch of images from the test dataset
        for test_batch in data.take(1):
            test_batch = test_batch[:num_samples]

            results = self.predict(test_batch)

            return results, test_batch


def sampling(args):
    z_mean, z_log_var, latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon


def load_dataset(dir_path):
    processed_images = []
    img_paths = [f for f in glob.iglob(os.path.join(dir_path, "*"))]

    for img_path in tqdm(img_paths):
        # normalize input
        img = nib.load(img_path)
        img_data = img.get_fdata()
        if img_data.shape[0] < img_data.shape[-1]:
            img_data = np.transpose(img_data, (1, 2, 0))

        img_data = np.expand_dims(img_data, axis=-1)
        img_normalized = img_data / np.max(img_data)

        img_segmentation = img_normalized
        processed_images.append(img_path)

    processed_images = np.array(processed_images)

    train_images, test_images = train_test_split(processed_images, test_size=0.05, random_state=42)
    train_ds = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

    return train_ds, test_ds


def create_tensorboard(base_path):
    tensorboard_path = os.path.abspath(base_path + "/tensorboard_logs/")
    log_path = os.path.abspath(tensorboard_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # start TensorBoard webapp
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_path, "--host", "localhost", "--port", "6006"])
    try:
        url = tb.launch()
        print(f"Tensorflow listening intern on {url}")
    except:
        print("Tensorflow already running!")
    # Create a log directory with a timestamp
    return TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True,
                               write_images=True)


def learning_rate_scheduler(epoch, lr):
    if epoch == 3:
        lr = learning_rate_2
    return lr


def generate_samples_resnet(model_path, input_dir_path, result_path, input_shape, set_window_zero):
    vae = ResNetVAE(input_shape)
    input_shape = (1, input_shape[0], input_shape[1], input_shape[2], 1)
    vae.encoder.build(input_shape=input_shape)
    vae.decoder.build(input_shape=(1, vae.img_latent_dim, vae.img_latent_dim, vae.img_latent_dim_z, vae.z_dim))
    vae.built = True
    vae.load_weights(model_path)

    img_paths = [f for f in glob.iglob(os.path.join(input_dir_path, "*"))]

    for img_path in tqdm(img_paths):
        # Normalize the image
        base, ext = os.path.splitext(img_path)
        index = os.path.basename(base)

        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_spacing = img.affine

        if img_data.shape[0] < img_data.shape[-1]:
            img_data = np.transpose(img_data, (1, 2, 0))

        img_data = np.expand_dims(img_data, axis=-1)
        img_data = img_data / np.max(img_data)
        img_data = np.expand_dims(img_data, axis=0)

        _img_data = vae.predict(img_data)
        _img_data = np.squeeze(_img_data)
        img_data = np.squeeze(img_data)

        print("ori_img: " + str(img_data.shape))
        print("syn_img: " + str(_img_data.shape))
        if set_window_zero:
            for i in range(img_data.shape[-1]):
                if np.all(img_data[:, :, i] == 0):
                    _img_data[:, :, i] = 0

        output_path = os.path.join(result_path, index + ".nii")
        nifti = nib.Nifti1Image(_img_data, img_spacing)
        nib.save(nifti, output_path)

