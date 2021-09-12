# also from irasin
#but i also revised this file for me to run the code
#tks for irasin

import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dataset
from dataset import *
import adain_model
import DDAdain_model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def gene(content, style, alpha, model_state_path, model, model_type):
   #OnLY for one img!直接把圖片丟進去，好像不用resize了
#     content = 'truely_want_content'#the dir of content img that you wanna transfer eg:./eg1/video/frame_0136.png
#     style = 'truely_want_style'#similat to above
#     output_name = '.final_synthesis'#Output path for generated image免副檔名
#     alpha = 1#alpha control the fusion degree in Adain[0, 1]
#     model_state_path = './result/model_state/???.pth'#save directory for result and loss(saved model state)
#model_type = which model? adain or DDAdain
    
    

    # set device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model
#     model = StyleTranserNetwork()
    if model_state_path is not None:
        model.load_state_dict(torch.load(model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(content)#RGB
    s = Image.open(style)#RGB
    c_tensor = trans(c).unsqueeze(0).to(device)#[1, 3, x, x]
    s_tensor = trans(s).unsqueeze(0).to(device)#[1, 3, x, x]

    with torch.no_grad(): 
        out = model.generate(c_tensor, s_tensor, alpha)
    
    out = denorm(out, device)

    output_name = os.path.basename(content)#jjj.png
    folder = content.split('/')[-3]
#         s_name = os.path.splitext(os.path.basename(style))[0]
#         output_name = f'{c_name}'
    if not (os.path.exists(f'./{folder}/styled_video_{model_type}')):
        os.mkdir(f'./{folder}/styled_video_{model_type}')#if content folder doesn't exist, create it
        
    save_image(out, f'./{folder}/styled_video_{model_type}/{output_name}', nrow=1)
    o = Image.open(f'./{folder}/styled_video_{model_type}/{output_name}')
    #new an image
    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)
    s = s.resize((i // 4 for i in c.size))

    demo.paste(c, (0, 0))#把c
    demo.paste(o, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    
    if not (os.path.exists(f'./{folder}/styled_video_no_compare_{model_type}')):
        os.mkdir(f'./{folder}/styled_video_no_compare_{model_type}')#if content folder doesn't exist, create it
    if not (os.path.exists(f'./{folder}/styled_video_compare_{model_type}')):
        os.mkdir(f'./{folder}/styled_video_compare_{model_type}')#if content folder doesn't exist, create it
        
    demo.save(f'./{folder}/styled_video_compare_{model_type}/{output_name}', quality=95)

    o.paste(s,  (0, o.height - s.height))
    o.save(f'./{folder}/styled_video_no_compare_{model_type}/{output_name}', quality=95)

    print(f'result saved into files starting with {output_name}')

class static_filename:
    #this class just used to update i
    #because we want to name a new file after using gene_imgs once
    i = 1
    
def gene_img(c_tensor, s_tensor, alpha, model1, model2, c_type):
# this function will do style transfer on content images with all style in the testset
# return some imgs

#     c = 'truely_want_content'#the dir of content img that you wanna transfer eg:./eg1/video/frame_0136.png
#     s = 'truely_want_style'#similat to above
#     alpha = 1#alpha control the fusion degree in Adain[0, 1]
#     c-type = 'person' or 'non_person'

    # set device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     c = Image.open(content)#RGB
#     s = Image.open(style)#RGB
#     c_tensor = trans(c).unsqueeze(0).to(device)#[1, 3, x, x]
#     s_tensor = trans(s).unsqueeze(0).to(device)#[1, 3, x, x]

    with torch.no_grad(): 
        out1 = model1.generate(c_tensor, s_tensor, alpha)
        out2 = model2.generate(c_tensor, s_tensor, alpha)
        
    out1 = denorm(out1, device)
    out2 = denorm(out2, device)
    style = denorm(s_tensor, device)
    
    
    save_path = './eg_images/synthesis/person' if c_type=='person' else './eg_images/synthesis/non_person'
    
#     out = [out1, out2]
    output_name = f'eg{static_filename.i}.png'#jjj.png
#     folder = content.split('/')[-3]
#         s_name = os.path.splitext(os.path.basename(style))[0]
#         output_name = f'{c_name}'
    out = torch.cat((out1, out2), dim=0)
    save_image(out, f'{save_path}/{output_name}', nrow=2)
    save_image(style, f'{save_path}/style.png', nrow=1)
    o = Image.open(f'{save_path}/{output_name}')
    s = Image.open(f'{save_path}/style.png')
    
    #set padding of style
    padding = 7
    style_pad = Image.new('RGB', (s.width + 2 * padding, s.height + 2 * padding))
    style_pad.paste(s, (padding, padding))
    ratio_s = style_pad.width/style_pad.height
    
    #new an image
    demo = Image.new('RGB', (o.width, o.height))
    
    s = style_pad.resize((int(ratio_s * (o.height//4)), o.height//4))
    

    demo.paste(o, (0, 0))#把c
    demo.paste(s, (o.width//2 - s.width//2, o.height - o.height//4))
        
    demo.save(f'{save_path}/{output_name}', quality=95)

    print(f'result saved into files starting with {output_name}')

# save the frame photos of video
def save_images(image, dir_path, imageid):
    img_path = dir_path + f'/frame_{imageid}.png'
    cv2.imwrite(img_path, image)

# read the frames of video
def read_video(video_path):
    '''
    convert initial video to the images
    and save it to the './egi/video' folder
    '''
    # read the video
    videoCapture = cv2.VideoCapture(video_path)
    # get the fps
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # read one frame
    success, frame = videoCapture.read()    # success is bool, telling whether the frame has been read successfully
    # videoCapture.read has rotate problem, we need to correct
#     frame = np.rot90(frame, 3)
    i = 0
    # loop until read ends, i is the imageid
    filename = os.path.split(video_path)[0]
    while success:
        if not os.path.exists(f'{filename}/video'):
            os.mkdir(f'{filename}/video')
        save_images(frame, f'{filename}/video', i)
        i = i + 1
        success, frame = videoCapture.read()
#         frame = np.rot90(frame, 3)

    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoCapture.release()
    
    return size

# To creat a video using frames
def create_video(dir_images, dir_video, fps):
    '''
    convert all the image in the folder to video
    
    input:(eg)
    dir_images:
    dir_video:
    size:size of image, is the return from read_video function
    '''
    frame = cv2.imread(f'{dir_images}/frame_1.png')
    sp = frame.shape
    print(sp)
    size = (sp[1], sp[0])
    # set the videoWriter
    videoWriter = cv2.VideoWriter(dir_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    filename = os.path.split(dir_video)[0]
    # read the framenames
    video_predictions = os.listdir(f'{filename}/video')
    # i is the imageid
    i = 0
    # create the video with frame
    while i <= len(video_predictions):
        frame_path = f'{dir_images}/frame_{i}.png'
        frame = cv2.imread(frame_path)
        videoWriter.write(frame)
        i = i + 1

def style_transfer(contents, style, alpha, model_state_path, model, model_type):
    '''
    this function will convert all images in contents folder to styled images
    input:(eg1)
    contents:'./eg1/video'
    style:'./eg1/style.jpeg'
    alpha:default is 1, this alpha will belongs to [0, 1]
            if alpha larger, the style will be stronger
    model_state_path:'./result/model_state_adain/4_epoch.pth'
    model:a class , maybe adain or DDAdain.
    model_type:'DDAdain' or 'adain'
    
    eg1
    |----------file-----------
    |-content.MOV
    |-content_styled_compare.MOV
    |-content_styled_no_compare.MOV
    |-content_styled.MOV
    |----------folder-----------
    |-video
    |-styled_video_no_compare_{model_type}
    |-styled_video_compare_{model_type}
    |-styled_video_{model_type}
    '''

    for content in glob.glob(contents+'/*'): 
        gene(content, style, alpha, model_state_path, model, model_type)
