import shutil
import os

src_dir = os.getcwd()
print(src_dir)

def Index():
    index = src_dir+"/Data/2015/val_image.txt" # change1
    index_list = []
    for i in open(index, 'r'):
        index_list.append(i.replace('\n', ''))
    return index_list

def CopyImage():
    save_dir = src_dir + "/Data/2015Image/ValImage" # change2
    j = 0
    for i in Index():
        # copy
        source_dir = src_dir+"/Data/twitter2015_images"
        source_file = os.path.join(source_dir, i) #文件夹+文件名  不能表示为指定路径
        shutil.copy(source_file, save_dir)

        # rename
        dst_file = os.path.join(save_dir, i)
        new_dst_file_name = os.path.join(save_dir, f'{j}img.jpg') # 避免数字重名覆盖丢失

        os.rename(dst_file, new_dst_file_name)
        os.chdir(save_dir)

        j += 1

if __name__ == '__main__':
    CopyImage()