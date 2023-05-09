import pathlib
import os
import json
import cv2

root = pathlib.Path('pytorch_faster_rcnn_tutorial/data/speed_bump')
ROOT_PATH = pathlib.Path(__file__).parent.absolute()

input_img0 = os.listdir(root / 'input2')[0]
source_dir_input = os.path.join(ROOT_PATH, root/'input2', input_img0)
image_shape = cv2.imread(source_dir_input).shape

targets = os.listdir(root / 'target2')
source_dir_tar = os.path.join(root, 'target2')

dest_dir_tar = os.path.join(root, 'target2_cxcywh')
pathlib.Path(dest_dir_tar).mkdir(parents=True, exist_ok=True)

mapping = {
    "speedbump": 1,
    "bumpsign": 2,
}

for i, filename in enumerate(targets):
    # полный путь к исходному файлу
    src_path = os.path.join(source_dir_tar, filename)
    with open(src_path, 'rb') as src_file:
        boxes = json.load(src_file)

    str_cxcywh =''
    for l, b in zip(boxes['labels'], boxes['boxes']):
        label = mapping[l]
        x1, y1, x2, y2 = b[0]/image_shape[1], b[1]/image_shape[0], b[2]/image_shape[1], b[3]/image_shape[0]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        str_cxcywh+="{:.0f}".format(label) + " " +\
            "{:.10f}".format(cx).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(cy).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(w).rstrip('0').rstrip('.') + " " +\
            "{:.10f}".format(h).rstrip('0').rstrip('.') + '\n'

    pre, ext = os.path.splitext(filename)
    dest_path_tar = os.path.join(dest_dir_tar, pre + '.txt')
    with open(dest_path_tar, 'w') as dest_file_tar:
        dest_file_tar.write(str_cxcywh)
        # print('str_cxcywh: ' , str_cxcywh)
        # input()
