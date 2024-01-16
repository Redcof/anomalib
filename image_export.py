import os
import pathlib
import random
import shutil
from collections import defaultdict
from typing import Iterable

import cv2
from tqdm import tqdm


def crawl(root, file_type=".hif"):
    """
    This is a file path generator for a given root directory.
    It crawls through all subdirectories recursively
    @param file_type:  The file type to search for
    @param root: The root directory
    @return:
    """
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(file_type):
                yield os.path.join(path, name)


def center_crop(img, h, w):
    """
    Crop image to the provided h and w, form the center of the image.
    where h < image.height, w < image.width
    """
    img_shape = img.shape
    x = img_shape[1] // 2 - w // 2
    y = img_shape[0] // 2 - h // 2
    
    crop_img = img[int(y):int(y + h), int(x):int(x + w)]
    return crop_img


def export(hif_path, save_path):
    """
    Read one HIF file and save each view as JPEG image in a given export directory as
    the following formal: HIF_NAME.hif-<view_index>.jpg,
    For example, HIF file: BAGGAGE_20231203_000000_12345.hif has 4 views, then the following files will be created:
        BAGGAGE_20231203_000000_12345.hif-0.jpg
        BAGGAGE_20231203_000000_12345.hif-1.jpg
        BAGGAGE_20231203_000000_12345.hif-2.jpg
        BAGGAGE_20231203_000000_12345.hif-3.jpg
    @param hif_path: The HIF path
    @param save_path: The path to export jpg images
    @return: None
    """
    from hif_reader.hif_file_reader import HifFileReader
    from hif_reader.tools.matlum_rgb import matlum_float_to_rgb
    src = os.path.join(save_path)
    path = os.path.join(src, "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    path = os.path.join(src, "train", "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    path = os.path.join(src, "test", "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    
    reader = HifFileReader()
    reader.read(hif_path, section_type_filter=["run_len_matlum"], )
    views_matlum = reader.get_sections_of_type("run_len_matlum")
    for view_idx, matlum in enumerate(views_matlum):
        channel_data = matlum["data"]["image_data_matlum_float"]
        rgb = matlum_float_to_rgb(channel_data)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        path = os.path.join(save_path, "%s-%d.jpg" % (os.path.basename(hif_path), view_idx))
        # center-crop to remove the trays forms the images
        if view_idx == 0:
            crop_h, crop_w = 590, 440
            rgb = center_crop(rgb, crop_h, crop_w)
        cv2.imwrite(path, rgb)
    return len(views_matlum)


def export_sd3_hif_form_dir(hif_path, img_export_path, remove_before_export=False):
    """
    Read all HIF files available in a directory and save each view as JPEG image in a given export directory as
    the following formal: HIF_NAME.hif-<view_index>.jpg,
    For example, HIF file: BAGGAGE_20231203_000000_12345.hif has 4 views, then the following files will be created:
        BAGGAGE_20231203_000000_12345.hif-0.jpg
        BAGGAGE_20231203_000000_12345.hif-1.jpg
        BAGGAGE_20231203_000000_12345.hif-2.jpg
        BAGGAGE_20231203_000000_12345.hif-3.jpg
    @param hif_path: The HIF path root directory
    @param img_export_path: The path to export jpg images
    @param remove_before_export: if set to True deletes all exported files, if any.
    @return: None
    """
    if remove_before_export:
        print("Removing...")
        shutil.rmtree(img_export_path)
    try:
        os.mkdir(img_export_path)
    except:
        ...
    print("Calculating files...", end="")
    files = list(crawl(hif_path))
    print("Done")
    print("Exporting...")
    counter = 0
    for path in tqdm(files):
        counter += export(path, img_export_path)
    print("\n%d images exported form %d HIF files" % (counter, len(files)))


def copy_files(src, files, dest, delete_at_source=False, overwrite=False):
    print("Moving" if delete_at_source else "Copying", end=" ")
    print("file form '%s' -> '%s'" % (src, dest))
    try:
        os.makedirs(dest)
    except:
        ...
    for file in tqdm(files):
        # `dest` needs to empty otherwise override needs to set to be True
        if os.path.exists(os.path.join(dest, file)) and not overwrite:
            continue
        # `src` needs it to be a file
        if os.path.isfile(os.path.join(src, file)):
            shutil.copyfile(os.path.join(src, file), os.path.join(dest, file))  # copy
            if delete_at_source:
                os.remove(os.path.join(src, file))  # remove copied
    print("")


def move_files(src, files, dest):
    copy_files(src, files, dest, delete_at_source=True)


def split_exported_sd3_hif(src, portion=.8, discard=.0):
    """
    This will create `train` and `test` directory and move images to two directories while following
    the portion given.
    For example, if we have 1000 images and portion=0.8, then the script will move 20%, 200 images
    to the newly created `test` subdirectory and 800 images to the `train` directory.
    @param src: the path where all exported jpg images is stored.
    @param portion: The percent to split. A float value between 0 and 1
    @param discard: discard files while splitting the dataset. this is used to reduce the dataset.
                    A decimal number [0, 1] indicates the percent portion of the images to discard.
                    An integer indicated the number of images to discard.
    @return: None
    """
    files = os.listdir(src)
    random.seed(47)
    random.shuffle(files)  # shuffle once
    random.seed(50)
    random.shuffle(files)  # shuffle twice
    assert discard > 0, "Invalid value {}".format(discard)
    if discard < 1:
        print("Discarding files... %.2d%% (%d files)" % (int(discard * 100), int(len(files) * discard)))
        files = files[:int(len(files) * (1. - discard))]
        files2del = files[int(len(files) * (1. - discard)):]
    else:
        print("Discarding files... %d" % int(discard))
        files = files[:int(discard)]
        files2del = files[int(discard):]
    first_half = files[:int(len(files) * portion)]
    second_half = files[int(len(files) * portion):]
    dest_train = os.path.join(src, "train")
    dest_test = os.path.join(src, "test")
    print("Splitting exported images")
    move_files(src, first_half, dest_train)
    move_files(src, second_half, dest_test)
    # delete unused files
    print("Deleting discarded files...")
    for file in tqdm(files2del):
        if os.path.isfile(os.path.join(src, file)): os.remove(os.path.join(src, file))


def copy_sixray_easy(sixray_path, anomaly_dataset_path, anomaly_classes):
    """
    This function will copy SIXray images to `train/0.normal`, `test/1.abnormal`, and `test/0.normal` directory.
    SIXray contains ~7000 positive images. Each image contains one/more threat classes. Using  `anomaly_classes`
    parameter user can decide which classes to be considered as abnormal and which are to be considered as
    normal.
    """
    sixray = pathlib.Path(sixray_path)
    src_train = sixray / "train" / "JPEGImages"
    src_test = sixray / "test" / "JPEGImages"
    # destination anomaly dataset dir
    anomaly = pathlib.Path(anomaly_dataset_path)
    sixray_train_files = list(crawl(src_train, ".jpg"))
    sixray_test_files = list(crawl(src_test, ".jpg"))
    # destination dirs.
    dest_train_normal = anomaly / "train" / "0.normal"  # stores normal train images
    dest_test_normal = anomaly / "test" / "0.normal"  # stores normal test images
    dest_test_abnormal = anomaly / "test" / "1.abnormal"  # stores abnormal train images
    
    if isinstance(anomaly_classes, Iterable):
        xml_train = sixray / "train" / "Annotations"  # contains sixray train annotations xmls
        xml_test = sixray / "test" / "Annotations"  # contains sixray test annotations xmls
        normal_train = []  # list for train images considered normal, will be stored in train normal dir.
        abnormal_train = []  # list for train images considered abnormal, will be stored in test abnormal dir.
        normal_test = []  # list for test images considered normal, will be stored in test normal dir.
        abnormal_test = []  # list for test images considered abnormal, will be stored in test abnormal dir.
        print("Segregating SIXray anomaly and normal files...")
        # loop for sixray train files
        for file in tqdm(sixray_train_files):
            xml = xml_train / os.path.basename(file).replace(".jpg", ".xml")
            with open(xml) as fp:
                s = fp.read()
                if any(map(lambda x: x in s, anomaly_classes)):
                    abnormal_train.append(file)
                else:
                    normal_train.append(file)
        # loop for sixray test files
        for file in tqdm(sixray_test_files):
            xml = xml_test / os.path.basename(file).replace(".jpg", ".xml")
            with open(xml) as fp:
                s = fp.read()
                if any(map(lambda x: x in s, anomaly_classes)):
                    abnormal_test.append(file)
                else:
                    normal_test.append(file)
        print("Copy Sixray data to anomaly dataset")
        # copying sixray/train/normal images to dataset/train/normal dir
        copy_files(src_train, list(map(os.path.basename, normal_train)), dest_train_normal)
        # copying sixray/train/abnormal images to dataset/test/abnormal dir
        copy_files(src_train, list(map(os.path.basename, abnormal_train)), dest_test_abnormal)
        # copying sixray/test/normal images to dataset/test/normal dir
        copy_files(src_test, list(map(os.path.basename, normal_test)), dest_test_normal)
        # copying sixray/train/abnormal images to dataset/test/abnormal dir
        copy_files(src_test, list(map(os.path.basename, abnormal_test)), dest_test_abnormal)
    else:
        print("Copy Sixray data to anomaly dataset")
        # copying sixray/train images to dataset/test/abnormal dir
        copy_files(src_train, list(map(os.path.basename, sixray_train_files)), dest_test_abnormal, overwrite=True)
        # copying sixray/test images to dataset/test/abnormal dir
        copy_files(src_test, list(map(os.path.basename, sixray_test_files)), dest_test_abnormal)


import xml.etree.ElementTree as ET


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract image information
    image_info = {'filename': root.find('filename').text, 'size': {
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'depth': int(root.find('size/depth').text)
    }}
    
    # Extract object information
    objects = []
    for obj in root.findall('object'):
        obj_info = {
            'name': obj.find('name').text,
            'bbox': {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text),
            }
        }
        objects.append(obj_info)
    
    return {
        'image': image_info,
        'objects': objects
    }


def classify_files(xml_train, sixray_train_files, anomaly_classes):
    """
    Prepare two dictionaries, for train and test respectively.
    The key to the dictionary is the class names. Each class
    name contains a list of file names.
    Args:
        xml_train:
        sixray_train_files:
        anomaly_classes:

    Returns:

    """
    normal_files = defaultdict(lambda: [])
    abnormal_files = defaultdict(lambda: [])
    for file in tqdm(sixray_train_files):
        xml = xml_train / os.path.basename(file).replace(".jpg", ".xml")
        # get the classnames
        classes = [d["name"] for d in parse_voc_xml(xml)['objects']]
        unique_cls = set(classes)
        if any(map(lambda x: x in unique_cls, anomaly_classes)):
            for cls in unique_cls:
                if cls in anomaly_classes:
                    abnormal_files[cls].append(file)
        else:
            for cls in unique_cls:
                normal_files[cls].append(file)
    return normal_files, abnormal_files


def copy_sixray_folder_split(sixray_path, anomaly_dataset_path, anomaly_classes):
    """
    This function will copy SIXray images to 'n' folders, where 'n' is number of classes. Each folder will be named
    by the class-name. This kind of structure is compatible with the 'anomalib' library.
    """
    assert isinstance(anomaly_classes, Iterable), "Provide a list of anomaly classes"
    sixray = pathlib.Path(sixray_path)
    src_train = sixray / "train" / "JPEGImages"
    src_test = sixray / "test" / "JPEGImages"
    # destination anomaly dataset dir
    anomaly_data_dir = pathlib.Path(anomaly_dataset_path)
    sixray_train_files = list(crawl(src_train, ".jpg"))
    sixray_test_files = list(crawl(src_test, ".jpg"))
    
    xml_train = sixray / "train" / "Annotations"  # contains sixray train annotations xmls
    xml_test = sixray / "test" / "Annotations"  # contains sixray test annotations xmls
    print("Segregating SIXray anomaly and normal files...")
    # loop for sixray train files
    normal_train, abnormal_train = classify_files(xml_train, sixray_train_files, anomaly_classes)
    normal_test, abnormal_test = classify_files(xml_test, sixray_test_files, anomaly_classes)
    
    print("Copy Sixray data to class folder data")
    # copying sixray/train/normal images to dataset/train/normal dir
    for cls, files in normal_train.items():
        copy_files(src_train, list(map(os.path.basename, files)), anomaly_data_dir / "normal" / cls)
    # copying sixray/train/abnormal images to dataset/test/abnormal dir
    for cls, files in abnormal_train.items():
        copy_files(src_train, list(map(os.path.basename, files)), anomaly_data_dir / "abnormal" / cls)
    # copying sixray/test/normal images to dataset/test/normal dir
    for cls, files in normal_test.items():
        copy_files(src_test, list(map(os.path.basename, files)), anomaly_data_dir / "normal" / cls)
    # copying sixray/train/abnormal images to dataset/test/abnormal dir
    for cls, files in abnormal_test.items():
        copy_files(src_test, list(map(os.path.basename, files)), anomaly_data_dir / "abnormal" / cls)


def clean_before_copy(anomaly_dataset_path):
    """Delete the anomaly dataset directory"""
    dataset_path = pathlib.Path(anomaly_dataset_path)
    shutil.rmtree(dataset_path, ignore_errors=True)


def copy_sd3_dataset(sd3_path, anomaly_dataset_path):
    """copy hif exported images to anomaly dataset."""
    sd3 = pathlib.Path(sd3_path)
    anomaly = pathlib.Path(anomaly_dataset_path)
    dest_train = anomaly / "train" / "0.normal"
    dest_test = anomaly / "test" / "0.normal"
    src_train = sd3 / "train"
    src_test = sd3 / "test"
    print("Copy SD3 data to anomaly dataset")
    copy_files(src_train, os.listdir(src_train), dest_train)
    copy_files(src_test, os.listdir(src_test), dest_test)


if __name__ == '__main__':
    sd3_hif_path = r"/data/bags_sd3"
    sixray_path = r"/data/Sixray_easy"
    sixray_sd3_anomaly_dataset = r"/data/sixray_sd3_anomaly"
    # sd3_hif_path = r"C:\Users\dndlssardar\OneDrive - Smiths Group\Documents\Projects\annox\example_images"
    sixray_path = r"C:\Users\dndlssardar\OneDrive - Smiths Group\Documents\Projects\Dataset\Sixray_easy"
    sixray_sd3_anomaly_dataset = r"C:\Users\dndlssardar\Documents\Projects\Dataset\Sixray_folder"
    sd3_img_export_path = os.path.join(sd3_hif_path, "exported")
    all_sixray_classes = ("gun", "knife", "wrench", "pliers", "scissors", "hammer")
    sixray_anomaly_classes = ("gun",)
    discard_sd3 = 30 / 100  # 30%
    
    clean_before_copy(sixray_sd3_anomaly_dataset)
    # export_sd3_hif_form_dir(sd3_hif_path, sd3_img_export_path, remove_before_export=True)
    # split_exported_sd3_hif(sd3_img_export_path, .8, discard=discard_sd3)
    # copy_sd3_dataset(sd3_img_export_path, sixray_sd3_anomaly_dataset)
    # copy_sixray_easy(sixray_path, sixray_sd3_anomaly_dataset, sixray_anomaly_classes)
    copy_sixray_folder_split(sixray_path, sixray_sd3_anomaly_dataset, sixray_anomaly_classes)
    print("Process completed.")
