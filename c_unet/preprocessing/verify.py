# Checks things like data integrity - comparing real data with things promised in json file etc.
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, join, isdir, isfile
import SimpleITK as sitk
from multiprocessing import Pool
import nibabel as nib


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()

    same_ori = np.all(np.isclose(ori1, ori2))
    if not same_ori:
        print("the origin does not match between the images:")
        print(ori1)
        print(ori2)

    same_spac = np.all(np.isclose(spacing1, spacing2))
    if not same_spac:
        print("the spacing does not match between the images")
        print(spacing1)
        print(spacing2)

    same_dir = np.all(np.isclose(direction1, direction2))
    if not same_dir:
        print("the direction does not match between the images")
        print(direction1)
        print(direction2)

    same_size = np.all(np.isclose(size1, size2))
    if not same_size:
        print("the size does not match between the images")
        print(size1)
        print(size2)

    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False


def verify_contains_only_expected_labels(itk_img: str, valid_labels: (tuple, list)):
    img_npy = sitk.GetArrayFromImage(sitk.ReadImage(itk_img))
    uniques = np.unique(img_npy)
    invalid_uniques = [i for i in uniques if i not in valid_labels]
    if len(invalid_uniques) == 0:
        r = True
    else:
        r = False
    return r, invalid_uniques


def verify_all_same_orientation(folder):
    """
    This should run after cropping
    :param folder:
    :return:
    """
    nii_files = subfiles(folder, suffix=".nii.gz", join=True)
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    # now we need to check whether they are all the same
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_dataset_integrity(folder, default_num_threads):
    """
    checks if all training cases and labels are present
    checks if all test cases (if any) are present
    for each case, checks whether all modalities are present
    for each case, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
    :param folder: folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    """
    assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
    assert isdir(join(folder, "imagesTr")), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
    assert isdir(join(folder, "labelsTr")), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder
    dataset = load_json(join(folder, "dataset.json"))
    training_cases = dataset['training']
    num_modalities = len(dataset['modality'].keys())
    test_cases = dataset['test']
    expected_train_identifiers = [i['image'].split("/")[-1][:-7] for i in training_cases]
    expected_test_identifiers = [i.split("/")[-1][:-7] for i in test_cases]

    # check training set
    nii_files_in_imagesTr = subfiles((join(folder, "imagesTr")), suffix=".nii.gz", join=False)
    nii_files_in_labelsTr = subfiles((join(folder, "labelsTr")), suffix=".nii.gz", join=False)

    label_files = []
    geometries_OK = True
    has_nan = False

    # check all cases
    if len(expected_train_identifiers) != len(np.unique(expected_train_identifiers)): raise RuntimeError("found duplicate training cases in dataset.json")

    print("Verifying training set")
    for c in expected_train_identifiers:
        print("checking case", c)
        # check if all files are present
        expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
        label_files.append(expected_label_file)
        expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
        assert isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (
            c, expected_label_file)
        assert all([isfile(i) for i in
                    expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
            c, expected_image_files)

        # verify that all modalities and the label have the same shape and geometry.
        label_itk = sitk.ReadImage(expected_label_file)

        nans_in_seg = np.any(np.isnan(sitk.GetArrayFromImage(label_itk)))
        has_nan = has_nan | nans_in_seg
        if nans_in_seg:
            print("There are NAN values in segmentation %s" % expected_label_file)

        images_itk = [sitk.ReadImage(i) for i in expected_image_files]
        for i, img in enumerate(images_itk):
            nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
            has_nan = has_nan | nans_in_image
            same_geometry = verify_same_geometry(img, label_itk)
            if not same_geometry:
                geometries_OK = False
                print("The geometry of the image %s does not match the geometry of the label file. The pixel arrays "
                      "will not be aligned and nnU-Net cannot use this data. Please make sure your image modalities "
                      "are coregistered and have the same geometry as the label" % expected_image_files[0][:-12])
            if nans_in_image:
                print("There are NAN values in image %s" % expected_image_files[i])

        # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
        for i in expected_image_files:
            nii_files_in_imagesTr.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    # check for stragglers
    assert len(
        nii_files_in_imagesTr) == 0, "there are training cases in imagesTr that are not listed in dataset.json: %s" % nii_files_in_imagesTr
    assert len(
        nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in dataset.json: %s" % nii_files_in_labelsTr

    # verify that only properly declared values are present in the labels
    print("Verifying label values")
    # print(dataset['labels'].keys())
    expected_labels = list(int(i) for i in dataset['labels'].keys())

    # check if labels are in consecutive order
    assert expected_labels[0] == 0, 'The first label must be 0 and maps to the background'
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    assert all(labels_valid_consecutive), f'Labels must be in consecutive order (0, 1, 2, ...). The labels {np.array(expected_labels)[1:][~labels_valid_consecutive]} do not satisfy this restriction'

    p = Pool(default_num_threads)
    results = p.starmap(verify_contains_only_expected_labels, zip(label_files, [expected_labels] * len(label_files)))
    p.close()
    p.join()

    fail = False
    print("Expected label values are", expected_labels)
    for i, r in enumerate(results):
        if not r[0]:
            print("Unexpected labels found in file %s. Found these unexpected values (they should not be there) %s" % (
                label_files[i], r[1]))
            fail = True

    if fail:
        raise AssertionError(
            "Found unexpected labels in the training dataset. Please correct that or adjust your dataset.json accordingly")
    else:
        print("Labels OK")

    # check test set, but only if there actually is a test set
    if len(expected_test_identifiers) > 0:
        print("Verifying test set")
        nii_files_in_imagesTs = subfiles((join(folder, "imagesTs")), suffix=".nii.gz", join=False)

        for c in expected_test_identifiers:
            # check if all files are present
            expected_image_files = [join(folder, "imagesTs", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
            assert all([isfile(i) for i in
                        expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
                c, expected_image_files)

            # verify that all modalities and the label have the same geometry. We use the affine for this
            if num_modalities > 1:
                images_itk = [sitk.ReadImage(i) for i in expected_image_files]
                reference_img = images_itk[0]

                for i, img in enumerate(images_itk[1:]):
                    assert verify_same_geometry(img, reference_img), "The modalities of the image %s do not seem to be " \
                                                                     "registered. Please coregister your modalities." % (
                                                                         expected_image_files[i])

            # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
            for i in expected_image_files:
                nii_files_in_imagesTs.remove(os.path.basename(i))
        assert len(
            nii_files_in_imagesTs) == 0, "there are training cases in imagesTs that are not listed in dataset.json: %s" % nii_files_in_imagesTr

    all_same, unique_orientations = verify_all_same_orientation(join(folder, "imagesTr"))
    if not all_same:
        print(
            "WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you correct that by reorienting the data. fslreorient2std should do the trick")
    # save unique orientations to dataset.json
    if not geometries_OK:
        raise Warning("GEOMETRY MISMATCH FOUND! CHECK THE TEXT OUTPUT! This does not cause an error at this point  but you should definitely check whether your geometries are alright!")
    else:
        print("Dataset OK")

    if has_nan:
        raise RuntimeError("Some images have nan values in them. This will break the training. See text output above to see which ones")
