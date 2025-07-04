import os
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.ndimage import zoom, center_of_mass
from tqdm import tqdm
import plotly.express as ex
from scipy.spatial import distance
from utils.helpers import dataset_loader, Sample, folder_loader
import cv2
mode = 0o777

def match_and_fusion_anomalies(control_samples_path, generated_samples_path, config,
                               output_dir, labels_output_dir, dublicate_factor=1):
    # collect information of control_samples
    temp_data = {}
    matching_data = []
    for control_sample in tqdm(folder_loader(control_samples_path)):
        if control_sample.array.shape in temp_data.keys():
            temp_data[control_sample.array.shape].append(control_sample.path)
        else:
            temp_data[control_sample.array.shape] = [control_sample.path]

    for _ in range(0, dublicate_factor):
        for syn in tqdm(folder_loader(generated_samples_path)):
            filename = syn.basename[:-4]
            ori_shape = config.syn_anomaly_transformations[filename]["shape"]

            shape = None
            low_distance = float('inf')

            for t in temp_data.keys():
                tmp_distance = distance.euclidean(t, ori_shape)
                if tmp_distance < low_distance:
                    low_distance = tmp_distance
                    shape = t

            control_sample_path = temp_data[shape][0]
            temp_data[shape].pop(0)
            if not temp_data[shape]:
                del temp_data[shape]

            matching_data.append([filename, os.path.basename(control_sample_path)])

            img = Sample(control_sample_path)

            fusion_anomaly(syn, img, output_dir, labels_output_dir,
                           config.syn_anomaly_transformations[filename]["scale_factor"],
                           config.syn_anomaly_transformations[filename]["centroid"])

    config.path_to_matching_infos = os.path.join(config.path_to_syn_run_folder, "matching_infos.csv")
    df_detection = pd.DataFrame(matching_data, columns=["anomaly", "control"])
    df_detection.to_csv(config.path_to_matching_infos, sep=',', encoding='utf-8', index=False)

    config.write_config_file()


def resize_and_pad(array, target_size=(96, 96, 32)):
    # Calculate the scale factors while maintaining aspect ratio
    print(array.shape)
    scale_factors = [min(t / s, 1) for s, t in zip(array.shape, target_size)]
    scaled_size = [int(s * f) for s, f in zip(array.shape, scale_factors)]

    # Scale down if necessary
    if any(scale_factors[i] < 1 for i in range(3)):
        array = zoom(array, scale_factors, order=1)  # order=1 for bilinear interpolation

    # Padding sizes for each dimension
    pad_widths = [((t - s) // 2, (t - s) - (t - s) // 2) for s, t in zip(array.shape, target_size)]

    # Apply padding
    array_padded = np.pad(array, pad_widths, mode='constant', constant_values=0)

    return array_padded, scale_factors


def crop_and_center_anomaly(seg_dir, img_dir, output_dir, config, target_size):
    from scipy.ndimage import label, find_objects
    for img, seg in dataset_loader(img_dir, seg_dir):
        _index = os.path.basename(img.path)
        index = _index[:-4]

        if np.all(seg.array == 0) or (seg.array is None):
            continue

        if img.array.shape[0] < img.array.shape[-1]:
            img.array = np.transpose(img.array, (1, 2, 0))

        if seg.array.shape[0] < seg.array.shape[-1]:
            seg.array = np.transpose(seg.array, (1, 2, 0))

        shape = img.array.shape

        print(img.array.shape)
        print(seg.array.shape)

        zero_indices = np.where(seg.array == 0)
        img.array[zero_indices] = 0
        binary_array = (img.array != 0).astype(int)
        labeled_array, num_features = label(binary_array)
        regions = find_objects(labeled_array)
        i = 0
        ori_image = img.array
        for region in regions:
            result = img.array[region]
            tmp = np.zeros(img.array.shape)
            tmp[region] = 1
            centroid = center_of_mass(tmp)
            centroid = tuple(round(ele1,2) / round(ele2,2) for ele1, ele2 in zip(centroid, tmp.shape))
            padded_arr, scale_factor = resize_and_pad(result, target_size=target_size)

            scale_factor = tuple(round(ele, 2) for ele in scale_factor)

            label_tmp = np.max(seg.array).round(0)
            config.add_anomaly_transformation(index + "_" + str(i), label_tmp, scale_factor, centroid, shape)
            output_path = os.path.join(output_dir, index + "_" + str(i) + ".nii")
            img.save_image(padded_arr, output_path)
            img.array = ori_image
            i += 1
        config.write_config_file()


def fusion_anomaly(anomaly, img, output_dir, seg_dir, scale_factor, position_factor):

    anomaly.array = np.where(anomaly.array > 0.01, anomaly.array, 0)
    # Normalize the pixel values to be between 0 and 255 (for uint8 encoding)
    min_val = np.nanmin(anomaly.array)
    max_val = np.nanmax(anomaly.array)

    # Handle division by zero or NaN values
    if min_val == max_val:
        anomaly.array = np.zeros_like(anomaly.array)
    else:
        anomaly.array = ((anomaly.array - min_val) / (max_val - min_val) * 255).astype('uint8')

    min_val = np.nanmin(img.array)
    max_val = np.nanmax(img.array)

    if min_val == max_val:
        img.array = np.zeros_like(img.array)
    else:
        img.array = ((img.array - min_val) / (max_val - min_val) * 255).astype('uint8')

    anomaly.array = trim_zeros(anomaly.array)

    if img.array.shape[0] < img.array.shape[-1]:
        img.array = np.transpose(img.array, (1, 2, 0))

    if anomaly.array.shape[0] < anomaly.array.shape[-1]:
        anomaly.array = np.transpose(anomaly.array, (1, 2, 0))

    anomaly.array = zoom(anomaly.array, scale_factor, order=1)

    offset = np.array((round((img.array.shape[0] * position_factor[0] - (anomaly.array.shape[0] / 2))),
                       round((img.array.shape[1] * position_factor[1] - (anomaly.array.shape[1] / 2))),
                       round((img.array.shape[2] * position_factor[2] - (anomaly.array.shape[2] / 2)))))
    if offset[2] < 0:
        offset[2] = 0

    offset_end = np.array((
        offset[0] + anomaly.array.shape[0], offset[1] + anomaly.array.shape[1], offset[2] + anomaly.array.shape[2]))

    anomaly_fusioned = np.where(anomaly.array > 0, anomaly.array,
                                img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]])

    img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]] = anomaly_fusioned

    offset = np.array((round((img.array.shape[0] * position_factor[0] - (anomaly.array.shape[0] / 2))),
                       round((img.array.shape[1] * position_factor[1] - (anomaly.array.shape[1] / 2))),
                       round((img.array.shape[2] * position_factor[2] - (anomaly.array.shape[2] / 2)))))
    if offset[2] < 0:
        offset[2] = 0

    offset_end = np.array((
        offset[0] + anomaly.array.shape[0], offset[1] + anomaly.array.shape[1], offset[2] + anomaly.array.shape[2]))

    for i, (o, e, s) in enumerate(zip(offset, offset_end, img.array.shape)):
        if e > s:
            offset[i] -= e - s
            offset_end[i] -= e - s

    img_data_tmp = img.array.copy()

    # normalising local
    img_region_max = np.nanmax(img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]])
    img_region_min = np.nanmin(img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]])

    anomaly_max = np.nanmax(anomaly.array)
    anomaly_min = np.nanmin(anomaly.array)

    _max = img_region_max / anomaly_max
    _min = img_region_min / anomaly_min

    anomaly.array = ((anomaly.array - anomaly_min) / (anomaly_max - anomaly_min) * np.nanmax(img_region_max)).astype(
        'uint8')

    background_threshold = 5
    index = img.path[-17:-4]
    seg = Sample(os.path.join(seg_dir, index + "fusioned.nii"))
    seg.array = np.zeros(img.array.shape)
    seg.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]] = np.where(
        anomaly.array > background_threshold, 1, 0)
    seg.array = seg.array.astype("uint8")

    anomaly_fusioned = np.where(anomaly.array > background_threshold, anomaly.array,
                                img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]])

    img.array[offset[0]:offset_end[0], offset[1]:offset_end[1], offset[2]:offset_end[2]] = anomaly_fusioned

    # this section fade anomaly contours into the sample

    tmp_blur = np.zeros((seg.array.shape[0], seg.array.shape[1]), np.uint8)
    tmp_combine = np.zeros((seg.array.shape[0], seg.array.shape[1]), np.uint8)

    tmp_blur_final = np.zeros(seg.array.shape, np.uint8)
    tmp_combine_final = np.zeros(seg.array.shape, np.uint8)

    for i in range(seg.array.shape[2]):
        if not np.all(seg.array[:, :, i] == 0):
            contours, hierarchy = cv2.findContours(np.uint8(seg.array[:, :, i]), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tmp_blur, contours, -1, 255, 1)
            cv2.drawContours(tmp_combine, contours, -1, 255, 3)

            fig = ex.imshow(tmp_blur)
            fig.show()

            print(tmp_combine.shape)
            print(tmp_combine_final[:, :, i].shape)

            tmp_combine_final[:, :, i] = tmp_combine
            tmp_blur_final[:, :, i] = tmp_blur

            tmp_blur = np.zeros((seg.array.shape[0], seg.array.shape[1]), np.uint8)
            tmp_combine = np.zeros((seg.array.shape[0], seg.array.shape[1]), np.uint8)

    img_data_tmp = (img_data_tmp + img.array) / 2

    combine_img = np.where(tmp_combine_final == 255, img_data_tmp, img.array)
    img.array = np.where(tmp_combine_final == 255, combine_img, img.array)

    blurred_img = scipy.ndimage.gaussian_filter(img.array, 0)
    img.array = np.where(tmp_blur_final == 255, blurred_img, img.array)

    if img.array.shape[-1] > img.array.shape[0]:
        img.array = np.transpose(img.array, (2, 1, 0))

    if seg.array.shape[-1] > seg.array.shape[0]:
        seg.array = np.transpose(seg.array, (2, 1, 0))

    seg.save_image()
    img.save_image(updated_path=os.path.join(output_dir, index + "fusioned.nii"))

    return


def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]
