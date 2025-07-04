import math
import os
import shutil
from os import listdir
from os.path import isfile, join
import seaborn as sns
import miseval
import nilearn
import nilearn.image
import numpy as np
import glob
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from utils.helpers import dataset_loader, folder_loader

mode = 0o777

def copy_samples_in_one_folder(src_dir, target_dir, recursive=True, modalities=None, clean_target=True):
    print()
    print("Copy images from source")
    if clean_target:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir, mode)
    if recursive:
        sample_paths = [f for f in glob.iglob(src_dir + '**/**', recursive=True) if isfile(join(src_dir, f))]
    else:
        sample_paths = [f for f in glob.iglob(os.path.join(src_dir, "*"))]
    for img_path in tqdm(sample_paths):
        if modalities is not None and img_path[-7:-4] not in modalities:
            continue
        filename = os.path.basename(img_path)
        target_path = os.path.join(target_dir, filename)
        shutil.copyfile(img_path, target_path)


def remove_samples_with_no_seg(img_dir, seg_dir):
    print()
    print("Remove samples with no segmentation")
    for img, seg in tqdm(dataset_loader(img_dir, seg_dir)):
        if (seg is None) and (img is not None):
            os.remove(img.path)


def scale_labels_to_img(img_dir, seg_dir):
    for img, seg in tqdm(dataset_loader(img_dir, seg_dir)):
        if img.array.shape != seg.array.shape:
            _tmp_seg = nilearn.image.resample_img(seg.obj, img.affine, img.array.shape)
            nib.save(_tmp_seg, seg.path)


def copy_labels(label_dir, target_dir, ai_label_dir=None, clean_target=True):
    print()
    print("Copy labels from source")
    if clean_target:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir, mode)
    for filename in tqdm(glob.glob(os.path.join(label_dir, '*'))):
        shutil.copy(filename, target_dir)

    if ai_label_dir is not None:
        direct_copy_counter = 0
        indirect_copy_counter = 0
        allready_added = os.listdir(target_dir)
        ai_paths = [f for f in glob.iglob(os.path.join(ai_label_dir, "*"))]
        for seg_path in tqdm(ai_paths):
            filename = os.path.basename(seg_path)
            if filename not in allready_added:
                direct_copy_counter += 1
                shutil.copy(seg_path, os.path.join(target_dir, filename))
            else:
                human_label_path = os.path.join(label_dir, filename)
                _human_seg = nib.load(human_label_path)
                _human_seg_data = _human_seg.get_fdata()
                unique_data = np.unique(_human_seg_data)
                index = unique_data[-1]
                if index > 0:
                    continue
                else:
                    _ai_seg = nib.load(seg_path)
                    _ai_seg_data = _ai_seg.get_fdata()
                    unique_data = np.unique(_ai_seg_data)
                    index = unique_data[-1]
                    if index > 0:
                        shutil.copy(seg_path, os.path.join(target_dir, filename))
                        indirect_copy_counter += 1
                    else:
                        continue

        print("AI Segmentations added")
        print("AI Segmentation added without Human Segmentation: " + str(direct_copy_counter))
        print("AI Segmentation replaced empty Human Segmentation: " + str(indirect_copy_counter))


def create_label_csv_from_labeldir(label_dir, output_path, superclass=False):
    print()
    print("Create CSV label list")
    label_paths = [f for f in glob.iglob(os.path.join(label_dir, "*"))]
    temp_data = []
    amount_of_seg_samples = 0
    for seg in tqdm(folder_loader(label_dir)):
        filename = seg.basename
        unique_data = np.unique(seg.array)
        index = np.max(unique_data)
        if superclass and index > 0:
            index = 1
            amount_of_seg_samples += 1
        if not superclass and index > 0:
            index = int(index) - 1
            amount_of_seg_samples += 1
        temp_data.append([filename, index])
    df_detection = pd.DataFrame(temp_data, columns=["SAMPLE", "CLASS"])
    df_detection.to_csv(output_path, sep=',', encoding='utf-8', index=False)
    print("CSV created")
    print("Number of Samples: " + str(len(label_paths)))
    print("Number of segmented Samples: " + str(amount_of_seg_samples))
    print("Number of control Samples: " + str(len(label_paths) - amount_of_seg_samples))


def convert_mha_to_nii(path_to_image, img_dir, image_name, remove_modality_index=False, remove=True):
    img = sitk.ReadImage(path_to_image)
    if remove:
        os.remove(path_to_image)
    if remove_modality_index:
        image_name = image_name[:-4]
    output_path = os.path.join(img_dir, image_name + ".nii")
    sitk.WriteImage(img, output_path)
    return output_path


def convert_dir_to_nii(img_dir, remove_modality_index=False):
    sample_paths = [f for f in glob.iglob(os.path.join(img_dir, "*"))]
    print()
    print("Convert complete dir to .nii")
    for sample in tqdm(sample_paths):
        filename = os.path.basename(sample)
        index = filename[:17]
        convert_mha_to_nii(sample, img_dir, index, remove_modality_index)


def convert_gz_to_nii(seg_dir):
    sample_paths = [f for f in glob.iglob(os.path.join(seg_dir, "*"))]
    print("Convert complete .gz dir to .nii")
    for sample in tqdm(sample_paths):
        filepath = sample[:-3]
        img = sitk.ReadImage(filepath)
        sitk.WriteImage(img, filepath)
        os.remove(sample)


def remove_control_samples(img_dir, label_dir):
    print()
    print("Remove Control Samples")
    for img, seg in tqdm(dataset_loader(img_dir, label_dir)):
        if seg.path is None:
            os.remove(img.path)
            continue
        if len(np.unique(seg.array)) <= 1:
            os.remove(img.path)
            os.remove(seg.path)


def copy_control_samples(img_dir, label_dir, target_dir):
    print()
    print("Copy Control Samples")
    for img, seg in tqdm(dataset_loader(img_dir, label_dir)):
        if len(np.unique(seg.array)) <= 1:
            filename = os.path.basename(img.path)
            target_path = os.path.join(target_dir, filename)
            shutil.copyfile(img.path, target_path)


def seperate_testdata(sample_dir, label_dir, img_output_dir, label_output_dir, percentige):
    # collect labeled samples
    amount_of_seg_samples = 0
    amount_of_controll_samples = 0
    for img, seg in tqdm(dataset_loader(sample_dir, label_dir)):
        unique_data = np.unique(seg.array)
        index = np.max(unique_data)
        if index > 0 and amount_of_seg_samples < percentige[0]:
            amount_of_seg_samples += 1
            shutil.copy(img.path, img_output_dir)
            shutil.copy(seg.path, label_output_dir)
            os.remove(img.path)
            os.remove(seg.path)
        if index == 0 and amount_of_controll_samples < percentige[1]:
            amount_of_controll_samples += 1
            shutil.copy(img.path, img_output_dir)
            shutil.copy(seg.path, label_output_dir)
            os.remove(img.path)
            os.remove(seg.path)
        if amount_of_seg_samples >= percentige[0] and amount_of_controll_samples >= percentige[1]:
            break


def normalize_label(seg_dir):
    for _, seg in tqdm(dataset_loader(None, seg_dir)):
        seg.array = np.where(seg.array > 1, 1, 0)
        seg.save_image()


def eval_model(pred_dir, seg_dir, output_dir, eval_metrics=None):

    if eval_metrics is None:
        eval_metrics = ["IoU", "MCC", "Dice", "AUC"]

    tmp_list = []
    for seg, pred in tqdm(dataset_loader(seg_dir, pred_dir)):
        if pred is None or seg is None:
            continue
        seg_data = seg.array
        pred_data = pred.array

        pred_data = np.where(pred_data >= 1, 1, 0)
        seg_data = np.where(seg_data >= 1, 1, 0)
        if len(np.unique(seg_data)) < 2:
            continue
        print(np.unique(seg_data))
        print(np.unique(pred_data))

        tmp_eval_results = []

        for metric in eval_metrics:
            tmp_eval_results.append(
                miseval.evaluate(seg_data, pred_data, metric=metric))

        tmp_eval_results.insert(0, pred.basename)
        tmp_list.append(tmp_eval_results)

    # visualize all scores with boxplot
    _columns = ["sample"]
    _columns.extend(eval_metrics)
    df = pd.DataFrame(tmp_list, columns=_columns)
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=df, notch=True)
    ax = sns.stripplot(data=df, linewidth=2)
    ax.set_title("all in one")
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, "boxplot.png"))
    fig.clf()

    df.to_csv(os.path.join(output_dir, "all_scores.csv"), sep='\t', encoding='utf-8', index=False)


def check_labels(dir1, dir):
    for _, seg in tqdm(dataset_loader(dir1, dir)):
        if 2 in np.unique(seg.array):
            print("ERROR")
            print(np.unique(seg.array))
