import pandas as pd
import paramiko
import pathlib
import shutil

# def copy_file(hostname, port, username, password, src, dst):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     print (" Connecting to %s \n with username=%s... \n" %(hostname,username))
#     t = paramiko.Transport(hostname, port)
#     t.connect(username=username,password=password)
#     sftp = paramiko.SFTPClient.from_transport(t)
#     print ("Copying file: %s to path: %s" %(src, dst))
#     sftp.put(src, dst)
#     sftp.close()
#     t.close() 

# df = pd.read_csv('full-gt-ice.csv')
df = pd.read_csv('full-gt.csv')
list_117 = df[df['sign_class']=='5_20']['filename'].to_list()

# from pytorch_faster_rcnn_tutorial.utils import get_filenames_of_path
# directory = pathlib.Path('../../media/storage3/adasys/data/icevision/imgs/')
src_dir = pathlib.Path(r'C:\Users\iserg\rebels_code\Tmp\sign117\images')
dst_dir = pathlib.Path(r'C:\Users\iserg\rebels_code\Tmp\sign520\images-520')
# dir = '/1.txt'
# image_files = get_filenames_of_path(directory / 'input')

# local_dir = pathlib.Path(r'C:\\Users\\iserg\\rebels_code\\PyTorch-Object-Detection-Faster-RCNN-Tutorial\\img117')  # путь к папке на локальном компьютере
# local_dir = pathlib.Path('C:\img117\1.txt')

hostname = '185.185.58.125'  # IP-адрес сервера на Linux
username = 's_filimonov'  # имя пользователя на сервере
password = '9?+59#ol921pk'  # пароль пользователя на сервере
port = 22


for f in list_117:
    src = pathlib.Path(src_dir / f)
    dst = pathlib.Path(dst_dir / f)
    # copy_file(hostname, port, username, password, src, dst)
    try:
        shutil.copyfile(src, dst)
    except:
        pass