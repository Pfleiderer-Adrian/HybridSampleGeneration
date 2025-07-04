import glob
from numpy import uint32
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from fpdf import FPDF


def data_exploration(img_dir, seg_dir, output_dir, num_of_classes=5, with_seg=True, resampling_shape=None):
    print()
    print("Execute data exploration")
    img_paths = [f for f in glob.iglob(os.path.join(img_dir, '*'))]
    sample_data = {}
    if resampling_shape is not None:
        resampling_shape = [resampling_shape[1], resampling_shape[2], resampling_shape[0]]

    for img_path in tqdm(img_paths):
        # Sample loading
        img = sitk.ReadImage(img_path)
        filename = os.path.basename(img_path)
        index = filename[:13]
        print("test")
        if with_seg:
            seg_path = os.path.join(seg_dir, index + "*.nii.gz")
            if not os.path.exists(seg_path):
                continue
            else:
                seg = sitk.ReadImage(seg_path)

        if resampling_shape is not None:
            print(img.GetSpacing())
            print(img.GetSize())
            ratio = np.array(img.GetSpacing()) / np.array(resampling_shape)
            new_shape = list(np.floor(img.GetSize() * ratio).astype(int))
            print(new_shape)
            # Resample imaging data
            reference_image = sitk.Image(new_shape[0].item(), new_shape[1].item(), new_shape[2].item(), img.GetPixelIDValue())
            reference_image.SetSpacing(resampling_shape)
            reference_image.SetOrigin(img.GetOrigin())
            reference_image.SetDirection(img.GetDirection())
            reference_seg = sitk.Image(new_shape[0].item(), new_shape[1].item(), new_shape[2].item(), seg.GetPixelIDValue())
            reference_seg.SetSpacing(resampling_shape)
            reference_seg.SetOrigin(seg.GetOrigin())
            reference_seg.SetDirection(seg.GetDirection())
            img = sitk.Resample(img, reference_image)
            seg = sitk.Resample(seg, reference_seg)

        spacing = img.GetSpacing()
        print(spacing)
        print(img.GetSize())
        spacing = (spacing[2], spacing[0], spacing[1])
        print()

        img_data = sitk.GetArrayFromImage(img)

        if with_seg:
            seg_data = sitk.GetArrayFromImage(seg)

        # Create an empty list for the current sample in our data dictionary
        sample_data[index] = []
        # Store the volume shape
        sample_data[index].append(img_data.shape)
        # Identify minimum and maximum volume intensity
        sample_data[index].append(np.min(img_data))
        sample_data[index].append(np.max(img_data))
        sample_data[index].append(np.mean(img_data))
        sample_data[index].append(np.median(img_data))
        # Store voxel spacing
        sample_data[index].append(spacing)
        # Store origin
        sample_data[index].append(img.GetOrigin())
        # Identify and store class distribution
        if with_seg:
            unique_data, unique_counts = np.unique(seg_data, return_counts=True)

            unique_sum = []
            class_appearance = []
            absolut_class_sum = []

            for i in range(0, num_of_classes+1):
                unique_sum.append(0)
                class_appearance.append(0)
                absolut_class_sum.append(0)
                if i in unique_data:
                    class_appearance[i] = 1
                    absolut_class_sum[i] = int(unique_counts[np.where(unique_data == i)])
                    unique_sum[i] = np.around(int(unique_counts[np.where(unique_data == i)]) / img_data.size, decimals=8)
            sample_data[index].append(tuple(unique_sum))
            sample_data[index].append(tuple(class_appearance))
            sample_data[index].append(tuple(absolut_class_sum))


    # Transform collected data into a pandas dataframe
    if with_seg:
        df = pd.DataFrame.from_dict(sample_data, orient="index",
                                    columns=["vol_shape", "vol_minimum",
                                             "vol_maximum", "vol_mean", "vol_median", "voxel_spacing", "origin",
                                             "class_frequency", "class_appearance", "class_absolut"])
    else:
        df = pd.DataFrame.from_dict(sample_data, orient="index",
                                    columns=["vol_shape", "vol_minimum",
                                             "vol_maximum", "vol_mean", "vol_median", "voxel_spacing", "origin"])



    df.to_csv(os.path.join(output_dir, "exploration_samplewise.csv"), sep='\t', encoding='utf-8',
              index=False)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    line_index = 1
    pdf.cell(200, 10, txt="Data Exploration", ln=line_index, align="C")
    line_index += 1
    if resampling_shape is not None:
        pdf.cell(200, 10, txt="(with Resampling: " + str(resampling_shape) + ")", ln=line_index, align="C")
        line_index += 1
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="----------------------------------------------------------------------------", ln=line_index, align="C")
    line_index += 1

    # Calculate mean and median shape sizes
    pdf.cell(200, 10, txt="Shape", ln=line_index, align="C")
    line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1
    shape_list = np.array(df["vol_shape"].tolist())
    for i, a in enumerate(["Z", "X", "Y"]):
        print(str(shape_list[:, i]))
        pdf.cell(200, 8, txt=a + "-Axes Shape Mean: " + str(np.mean(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Shape Median: " + str(np.median(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Shape Min: " + str(np.min(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Shape Max: " + str(np.max(shape_list[:, i])), ln=2, align="L")
        line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1

    # Calculate mean and median shape sizes
    pdf.cell(200, 10, txt="Origin", ln=line_index, align="C")
    line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1
    shape_list = np.array(df["vol_shape"].tolist())
    for i, a in enumerate(["Z", "X", "Y"]):
        pdf.cell(200, 8, txt=a + "-Axes Origin Mean: " + str(np.mean(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Origin Median: " + str(np.median(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Origin Min: " + str(np.min(shape_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Origin Max: " + str(np.max(shape_list[:, i])), ln=2, align="L")
        line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1

    pdf.cell(200, 10, txt="Pixel Value", ln=line_index, align="C")
    line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1
    pdf.cell(200, 8, txt="Mean Max Pixel Value: " + str(np.mean(np.array(df["vol_maximum"].tolist()))), ln=2, align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Median Max Pixel Value: " + str(np.median(np.array(df["vol_maximum"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Overall Max Pixel Value: " + str(np.max(np.array(df["vol_maximum"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Mean Min Pixel Value: " + str(np.mean(np.array(df["vol_minimum"].tolist()))), ln=2, align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Median Min Pixel Value: " + str(np.median(np.array(df["vol_minimum"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Overall Min Pixel Value: " + str(np.max(np.array(df["vol_minimum"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Mean Mean Pixel Value: " + str(np.mean(np.array(df["vol_mean"].tolist()))), ln=2, align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Median Mean Pixel Value: " + str(np.median(np.array(df["vol_mean"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Overall Mean Pixel Value: " + str(np.max(np.array(df["vol_mean"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Mean Median Pixel Value: " + str(np.mean(np.array(df["vol_median"].tolist()))), ln=2, align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Median Median Pixel Value: " + str(np.median(np.array(df["vol_median"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 8, txt="Overall Median Pixel Value: " + str(np.max(np.array(df["vol_median"].tolist()))), ln=2,
             align="L")
    line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1

    pdf.cell(200, 10, txt="Spacing", ln=line_index, align="C")
    line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1
    spacing_list = np.array(df["voxel_spacing"].tolist())
    for i, a in enumerate(["Z", "X", "Y"]):
        pdf.cell(200, 8, txt=a + "-Axes Spacing Mean: " + str(np.mean(spacing_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Spacing Median: " + str(np.median(spacing_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Spacing Min: " + str(np.min(spacing_list[:, i])), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt=a + "-Axes Spacing Max: " + str(np.max(spacing_list[:, i])), ln=2, align="L")
        line_index += 1
    pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
    line_index += 1

    if with_seg:

        pdf.cell(200, 10, txt="Class distribution", ln=line_index, align="C")
        line_index += 1
        pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
        line_index += 1
        pdf.cell(200, 8, txt="Number of Samples: " + str(len(img_paths)), ln=2, align="L")
        line_index += 1
        pdf.cell(200, 8, txt="Number of Classes: " + str(num_of_classes), ln=2, align="L")
        line_index += 1
        class_onehot = np.array(df["class_appearance"].tolist())
        absolut_list = np.array(df["class_absolut"].tolist())
        for i in range(0, num_of_classes+1):
            pdf.cell(200, 8, txt="No. of Samples of Seg. Label " + str(i) + ": " + str(np.sum(class_onehot[:, i])), ln=2,
                     align="L")
            line_index += 1
            pdf.cell(200, 8, txt="No. of Pixels of Seg. Label " + str(i) + ": " + str(np.sum(absolut_list[:, i])), ln=2,
                     align="L")
            line_index += 1
        pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
        line_index += 1

        pdf.cell(200, 10, txt="Class frequency", ln=line_index, align="C")
        line_index += 1
        pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
        line_index += 1
        frequency_list = np.array(df["class_frequency"].tolist())
        for i, a in enumerate(range(0, num_of_classes+1)):
            pdf.cell(200, 8, txt=str(a) + " Label Frequency Mean: " + str(np.around(np.mean(frequency_list[:, i]), decimals=8)), ln=2, align="L")
            line_index += 1
            pdf.cell(200, 8, txt=str(a) + " Label Frequency Median: " + str(np.around(np.median(frequency_list[:, i]), decimals=8)), ln=2, align="L")
            line_index += 1
            pdf.cell(200, 8, txt=str(a) + " Label Frequency Min: " + str(np.around(np.min(frequency_list[:, i]), decimals=8)), ln=2, align="L")
            line_index += 1
            pdf.cell(200, 8, txt=str(a) + " Label Frequency Max: " + str(np.around(np.max(frequency_list[:, i]), decimals=8)), ln=2, align="L")
            line_index += 1
        pdf.cell(200, 10, txt="--------------------------------------", ln=line_index, align="C")
        line_index += 1

    if resampling_shape is not None:
        pdf.output(os.path.join(output_dir, "exploration_summary(resampled).pdf"))
    else:
        pdf.output(os.path.join(output_dir, "exploration_summary.pdf"))


def calculate_max_shape(path_to_dir):
    img_paths = [f for f in glob.iglob(os.path.join(path_to_dir, '*'))]

    y_sizes = []
    x_sizes = []
    z_sizes = []

    for img_path in tqdm(img_paths):
        # Sample loading
        img = sitk.ReadImage(img_path)
        size = img.GetSize()
        x_sizes.append(size[0])
        y_sizes.append(size[1])
        z_sizes.append(size[2])

    print("MaxSize: " + str(np.max(x_sizes)) + " " + str(np.max(y_sizes)) + " " + str(np.max(z_sizes)))