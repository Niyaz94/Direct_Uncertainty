# Finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
import os
import pickle
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir, subfiles
import shutil
import json
from collections import OrderedDict


# MS imported in preprocessing.py
def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def create_lists_from_splitted_dataset(base_folder_splitted):
    """
    :param base_folder_splitted: task folder
    :return: list of the training files and dictionary with modalities number:name pairs (e. g. "0": "T2")
    """
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    # print(d['modality'])
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes  # MS: It does exactly what its name implies. https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties


def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        Finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        :param num_threads:
        :param output_folder: whete to store the cropped data
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)


def crop(override, num_threads, nnUNet_raw_data, nnUNet_cropped_data):
    """
    Creates an object of a ImageCropper class and then its methods to find a mask of nonzero elements (must be nonzero
    in all modalities) and crop the image to that mask. It is executed with threads Pool.
    :param task_string: task_name, e.g. Task007_Liver
    :param override: if True is set the current cropped folder with it contents will be deleted before creating a new one.
    :param num_threads: default_num_threads from configuration file (config_prep.json).
    """
    # Folder is taken from manually created folders.
    cropped_out_dir = nnUNet_cropped_data
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    # MS: Creating lists based on modality.
    splitted_4d_output_dir_task = nnUNet_raw_data
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # MS: Just a class instantiation.
    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    # MS: real action.
    imgcrop.run_cropping(lists, overwrite_existing=override)
    # MS: copying json file to the cropped folder.
    shutil.copy(join(nnUNet_raw_data, "dataset.json"), cropped_out_dir)
