path1 = './images'
path = './results'
file_list = os.listdir(path)
id = []
for i in range(len(file_list)):
    id.append(file_list[i].split('_')[1])
id = set(id)
sort_folder_number = list(id)
for n in sort_folder_number:
    if n[-3:] == 'png':
        n = n[:-4]
    new_folder_path = os.path.join(path1, '%s'%n)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
for i in range(len(file_list)):
    old_file_path = os.path.join(path, file_list[i])
    fid = file_list[i].split('_')[1]
    if fid[-3:] == 'png':
        fid = fid[:-4]
    new_file_path = os.path.join(path1, '%s' %(fid), file_list[i])
    shutil.move(old_file_path, new_file_path)
