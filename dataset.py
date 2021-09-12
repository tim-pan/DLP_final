
# this file is revised from https://github.com/irasin/Pytorch_AdaIN
#thanks @irasin! to save my final project!
import os
import glob
import numpy as np
import random
import shutil
from tqdm import tqdm
from skimage import io, transform
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms




trans = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)#reduce tensor to raw images
#     res = tensor * std + mean
    return res


def random_files_content(num):
    '''
    this function randomly choose num images from cocoset
    and check if there are some .ds-store or._ file in folder 
    and copy them to the 'content' folder simultaneously
    '''   
    list_dir = [f for f in os.listdir('./raw_content') if not f.startswith('.')]
    
    if not (os.path.exists('train_content')):
        os.mkdir('train_content')#if content folder doesn't exist, create it
    if not (os.path.exists('test_content')):
        os.mkdir('test_content')#if content folder doesn't exist, create it
        
    #randomly choose, num images from raw_content (had already REMOVED '.'files)
    list_dir = random.sample(list_dir, num)
    random.shuffle(list_dir)
    train_dirs = list_dir[:int(0.8*num)]
    test_dirs = list_dir[int(0.8*num)+1: num]
    
    for di in train_dirs:   
        shutil.copy(os.path.join('raw_content', di), './train_content/')#copy './raw_content/PIC.jpg to ./content/'
    for di in test_dirs:   
        shutil.copy(os.path.join('raw_content', di), './test_content/')#copy './raw_content/PIC.jpg to ./content/'

def random_files_style(num):
    all_images = []
    #put all images in raw stye to a list
    for root, dirs, files in os.walk('./raw_style'):#take all files in raw-style with full path
        for file in files:
            #remove ._ and .DS store
            if not file.startswith('.'):
                fullpath = os.path.join(root, file)
                all_images.append(fullpath)
                
    if not (os.path.exists('train_style')):
        os.mkdir('train_style')#if folder doesn't exist, create it
    if not (os.path.exists('test_style')):
        os.mkdir('test_style')#if folder doesn't exist, create it
    
    dirs = random.sample(all_images, num)   
    random.shuffle(dirs)
    train_dirs = dirs[:int(0.8*num)]
    test_dirs = dirs[int(0.8*num)+1: num]
    for di in train_dirs:   
        shutil.copy(di, './train_style/')#copy './raw_style/PIC.jpg to ./style/'
    for di in test_dirs:
        shutil.copy(di, './test_style/')#copy './raw_style/PIC.jpg to ./style/'
    
        
    
class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=trans):        
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'
        if not (os.path.exists(content_dir_resized) and
                os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)
            #resize the raw dataset and save it to content(style)_resized
            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)
        #and take all images in new folder
        content_images = glob.glob((content_dir_resized + '/*'))
        #and do shuffle
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + '/*')
        np.random.shuffle(style_images)
        
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    @staticmethod
    #this method try to resize all the pictures in the dataset, 
    #and save it in the content_resized and style_resized
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
#             print(i)
            filename = os.path.basename(i)
            try:
                image = io.imread(os.path.join(source_dir, i))
                #把擺得很奇怪的image轉回來
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
#                     print(H, W)
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                        
#                     print(H, W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
            except:
                continue

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        # content_image = io.imread(content_image, plugin='pil')
        # style_image = io.imread(style_image, plugin='pil')
        # Unfortunately,RandomCrop doesn't work with skimage.io
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return content_image, style_image
    
    
if __name__ == '__main__':
    num = 5000
    random_files_content(num)
    random_files_style(num)
# if your data doesn't split into train data 
# and test data please run this two files
    
    
    
    
    
