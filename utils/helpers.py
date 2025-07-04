import glob
import os
import shutil
import copy
import nibabel as nib
import numpy as np


def create_folder(path, clear_target=False):
    mode = 0o777
    if not os.path.exists(path):
        os.mkdir(path, mode)
        return True
    if clear_target:
        shutil.rmtree(path)
        os.mkdir(path, mode)
        return True
    return False


class dir_loader:
    def __iter__(self, path):
        self.paths = [f for f in glob.iglob(os.path.join(path, "*"))]
        return self

    def __next__(self):
        _path = self.paths.pop()
        item = nib.load(_path)
        array = item.get_fdata()
        return array, _path


class dataset_loader:

    def __init__(self, img_dir, seg_dir):
        self.seg_dir = seg_dir
        self.img_dir = img_dir
        self.seg_paths = [f for f in glob.iglob(os.path.join(self.seg_dir, "*"))]
        self.img_paths = [f for f in glob.iglob(os.path.join(self.img_dir, "*"))]
        self.union_paths = []
        self.tmp_img_paths = copy.deepcopy(self.img_paths)
        self.tmp_seg_paths = copy.deepcopy(self.seg_paths)
        for img_path in self.img_paths:
            img_basename = os.path.basename(img_path)
            for seg_path in self.seg_paths:
                seg_basename = os.path.basename(seg_path)
                #if seg_basename == img_basename:
                if seg_basename[:-4]+"_0000.nii" == img_basename:
                    self.union_paths.append((img_path, seg_path))
                    self.tmp_img_paths.remove(img_path)
                    self.tmp_seg_paths.remove(seg_path)
        for img_path in self.tmp_img_paths:
            self.union_paths.append((img_path, None))
        for seg_path in self.tmp_seg_paths:
            self.union_paths.append((None, seg_path))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.union_paths:
            raise StopIteration
        _sample_path = self.union_paths.pop()
        _img_path = _sample_path[0]
        _seg_path = _sample_path[1]

        if _seg_path is not None:
            _seg = Sample(_seg_path)
            _seg.unload()
        else:
            _seg = None

        if _img_path is not None:
            _img = Sample(_img_path)
            _img.unload()
        else:
            _img = None
        return _img, _seg


class folder_loader:
    def __init__(self, folder_dir):
        self.folder_dir = folder_dir
        self.file_paths = [f for f in glob.iglob(os.path.join(self.folder_dir, "*"))]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.file_paths:
            raise StopIteration
        _file_path = self.file_paths.pop()
        if _file_path is not None:
            _file = Sample(_file_path)
        else:
            _file = None

        return _file


class Sample:
    def __init__(self, _path):
        self.path = _path
        self.basename = os.path.basename(_path)
        if os.path.exists(self.path):
            self.obj = nib.load(self.path)
            self.array = self.obj.get_fdata()
            self.affine = self.obj.affine
            self.unload()
        else:
            self.obj = None
            self.array = None
            self.affine = None

    def save_image(self, updated_image=None, updated_path=None):
        if updated_image is not None:
            self.array = updated_image
        if updated_path is not None:
            self.path = updated_path
        tmp_obj = nib.Nifti1Image(self.array, self.affine, dtype=np.int32)
        nib.save(tmp_obj, self.path)

    def unload(self):
        self.obj.uncache()

