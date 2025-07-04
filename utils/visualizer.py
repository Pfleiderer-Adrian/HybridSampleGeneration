import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns


def plot_slices(images, original_images, slice_axis=2):
    num_images = images.shape[0] * 2
    num_slices = images.shape[slice_axis]
    tmp_index = 0
    plt.figure(figsize=(num_slices * 3, num_images * 3))

    for i, (img, img_o) in enumerate(zip(images, original_images)):
        for j in range(num_slices):
            # Calculate the index for the subplot
            subplot_index = i * num_slices + j + 1 + tmp_index

            # Extract the slice based on the specified axis
            if slice_axis == 0:
                slice_ = img[j, :, :]
                slice_o = img_o[j, :, :]
            elif slice_axis == 1:
                slice_ = img[:, j, :]
                slice_o = img_o[:, j, :]
            else:
                slice_ = img[:, :, j]
                slice_o = img_o[:, :, j]

            plt.subplot(num_images, num_slices, subplot_index)
            plt.imshow(slice_o, cmap='gray')
            plt.axis('off')
            plt.title(f'Img {i + 1}, Slice {j + 1}')
            plt.subplot(num_images, num_slices, subplot_index + num_slices)
            plt.imshow(slice_, cmap='gray')
            plt.axis('off')
            plt.title(f'Img {i + 1}, Slice {j + 1}')
        tmp_index += num_slices
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_sample(syn_image, ori_image, plot_axis, output_path):
    num_slices = ori_image.shape[plot_axis]
    num_images = 2
    plt.figure(figsize=(num_slices * 3, num_images * 3))
    for j in range(num_slices):
        # Extract the slice based on the specified axis
        # Calculate the index for the subplot
        if plot_axis == 0:
            slice_ = syn_image[j, :, :]
            slice_o = ori_image[j, :, :]
        elif plot_axis == 1:
            slice_ = syn_image[:, j, :]
            slice_o = ori_image[:, j, :]
        else:
            slice_ = syn_image[:, :, j]
            slice_o = ori_image[:, :, j]

        plt.subplot(num_images, num_slices, j + 1)
        plt.imshow(slice_o, cmap='gray')
        plt.axis('off')
        plt.title(f'Slice {j + 1}')
        plt.subplot(num_images, num_slices, j + 1 + num_slices)
        plt.imshow(slice_, cmap='gray')
        plt.axis('off')
        plt.title(f'Slice {j + 1}')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(output_path)


def overlay_segmentation(vol, seg, cm="hls", alpha=0.9):
    num_classes = len(np.unique(seg)) - 1

    # Convert volume to RGB
    vol = np.squeeze(vol)
    seg = np.squeeze(seg)

    vol_rgb = np.stack([vol, vol, vol], axis=-1)
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.int)

    # Get unique classes from the sample
    unique_classes, tmp = np.unique(seg, return_counts=True)
    # Get color palette for all classes
    color_palette = sns.color_palette(cm, num_classes)

    print(seg.shape)
    print(seg_rgb.shape)

    for i, label in np.ndenumerate(unique_classes):
        label = int(label)
        seg_rgb[np.equal(seg, label)] = (
            color_palette[label][0] * 255, color_palette[label][1] * 255, color_palette[label][2] * 255)

    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)

    # Weighted sum where there's a value to overlay
    vol_overlayed = np.where(
        repeated_segbin,
        np.round(alpha * seg_rgb + (1 - alpha) * vol_rgb).astype(np.uint8),
        np.round(vol_rgb).astype(np.uint8)
    )
    return vol_overlayed


def visualize_sample(img_data, seg_data, pred_data, out_dir, title=""):
    vol_truth = overlay_segmentation(img_data, seg_data)
    vol_pred = overlay_segmentation(img_data, pred_data)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Initialize the two subplots (axes) with an empty 512x512 image
    data = vol_truth[:, :, 0]
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    img1 = ax1.imshow(data)
    img2 = ax2.imshow(data)

    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + title + " - " + "Slice: " + str(i))
        img1.set_data(vol_truth[:, :, i])
        img2.set_data(vol_pred[:, :, i])
        return [img1, img2]

    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=vol_truth.shape[2], interval=200,
                                  repeat_delay=0, blit=False)

    file_name = "case_" + title.zfill(5) + ".gif"
    out_path = os.path.join(out_dir, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick')
    # Close the matplot
    plt.close()


def grayscale_normalization(image):
    # Identify minimum and maximum
    max_value = np.max(image)
    min_value = np.min(image)
    # Scaling
    image_scaled = (image - min_value) / (max_value - min_value)
    image_normalized = np.around(image_scaled * 255, decimals=0)
    # Return normalized image
    return image_normalized


def visualize_evaluation(case_id, volume, eva_path):
    # Grayscale Normalization of Volume
    volume_gray = grayscale_normalization(volume)
    # Create a figure and two axes objects from matplot
    fig = plt.figure()
    img = plt.imshow(volume_gray[0, :, :], cmap='gray', vmin=0, vmax=255,
                     animated=True)

    # Update function to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - " + "Slice: " + str(i))
        img.set_data(volume_gray[i, :, :])
        return img

    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=volume_gray.shape[0],
                                  interval=5, repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    file_name = "visualization." + str(case_id) + ".gif"
    out_path = os.path.join(eva_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=None, dpi=None)
    # Close the matplot
    plt.close()


def visualize_4D_evaluation(case_id, volume, eva_path, channel, seg=None):
    # Grayscale Normalization of Volume
    volume = np.squeeze(volume[:, :, :, channel])
    volume_gray = grayscale_normalization(volume)
    if seg is not None:
        volume_gray = overlay_segmentation(volume_gray, seg)

    # Create a figure and two axes objects from matplot
    fig = plt.figure()
    img = plt.imshow(volume_gray[:, :, 0], cmap='gray', vmin=0, vmax=255,
                     animated=True)

    # Update function to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - Channel" + str(channel) + " - " + "Slice: " + str(i))
        img.set_data(volume_gray[:, :, i])
        return img

    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=volume_gray.shape[2],
                                  interval=5, repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    file_name = "visualization." + str(case_id) + ".gif"
    out_path = os.path.join(eva_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=None, dpi=None)
    # Close the matplot
    plt.close()
