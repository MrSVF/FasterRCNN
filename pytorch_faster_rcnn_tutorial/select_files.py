import os
import random
import shutil

# путь к исходной папке
source_dir = 'pytorch_faster_rcnn_tutorial\data\speed_bump\set2'

# путь к папке, в которую будут сохранены выбранные файлы
dest_dir = 'pytorch_faster_rcnn_tutorial/data/speed_bump/input2'
dest_dir2 = 'pytorch_faster_rcnn_tutorial/data/speed_bump/test2'

# количество файлов для выборки
frac = 0.5

# получаем список файлов в исходной папке
files = os.listdir(source_dir)

# выбираем случайные файлы из списка
selected_files = random.sample(files, int(frac*len(files)))
selected_files2 = list(set(files) - set(selected_files))

# копируем выбранные файлы в папку назначения
for file_name in selected_files:
    file_path = os.path.join(source_dir, file_name)
    shutil.copy(file_path, dest_dir)

for file_name in selected_files2:
    file_path = os.path.join(source_dir, file_name)
    shutil.copy(file_path, dest_dir2)