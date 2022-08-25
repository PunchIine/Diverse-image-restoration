import os
import shutil
path1 = './images'
path = './results'
file_list = os.listdir(path)
id = ['out0', 'out1']
sort_folder_number = list(id)
for n in sort_folder_number:
    if n[-3:] == 'png':
        n = n[:-4]
        n = n + n[-5]
    new_folder_path = os.path.join(path1, '%s'%n)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
for i in range(len(file_list)):
    old_file_path = os.path.join(path, file_list[i])
    if file_list[i].split('_')[1] == 'truth.png':
        continue
    fid = file_list[i].split('_')[2]
    if fid[-3:] == 'png':
        fid = "out" + fid[:-4]
    new_file_path = os.path.join(path1, '%s' %(fid), file_list[i])
    shutil.move(old_file_path, new_file_path)
