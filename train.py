#@irasin
#train.py revised from train.py in the github of irasin
#because its too trouble to write a train.py again
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
import adain_model
import DDAdain_model


def main():
    '''hyperparameters'''
    batch_size = 4
    epoch = 40
    learning_rate = 5e-5
    train_content_dir = 'train_content'#raw data to splited data is written in dataset.py
    train_style_dir = 'train_style'#all these are splited data, but without resize
    test_content_dir = 'test_content'
    test_style_dir = 'test_style'
    save_dir = 'result'#loss, model_state, image will be saved in this file
    model_type = 'DDAdain'
    
    reuse = None#model state path to load
    
    snapshot_interval = 1
    #similar to print-every, it will show the styled pic after snapshot interval
    '''the end of setting hyperparameters'''

    # create directory to save
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    loss_dir = f'{save_dir}/loss_{model_type}'
    model_state_dir = f'{save_dir}/model_state_{model_type}'
    image_dir = f'{save_dir}/image_{model_type}'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print(f'# Minibatch-size: {batch_size}')
    print(f'# epoch: {epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(train_content_dir, train_style_dir)
    test_dataset = PreprocessDataset(test_content_dir, test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)
    test_iter = iter(test_loader)

    # set model and optimizer
    if model_type == 'adain':
        model = adain_model.StyleTranserNetwork().to(device)
    elif model_type == 'DDAdain':
        model = DDAdain_model.StyleTranserNetwork().to(device)
    else:
        raise WrongArgumentError
    
    if reuse is not None:
        model.load_state_dict(torch.load(reuse))
        
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # start training
    loss_list = []
    
    for e in range(1, epoch + 1):
        num = 0
        total_loss = 0
        total_c_loss = 0
        total_s_loss = 0
        for content, style in tqdm(train_loader):
            content = content.to(device)#(bs, c, h, w)
            style = style.to(device)#(bs, c, h, w)
            
            num += content.size(0)
            loss, c_loss, s_loss = model(content, style)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_s_loss += s_loss.item()
            num += 1
            
        print(f'epoch:{e}')
        print(f'c_loss:{(total_c_loss/num):.4f}, s_loss:{(total_s_loss/num):.4f}')
        print(f'train_loss:{(total_loss/num):.4f}')
        print()
        if e % snapshot_interval == 0:
            content, style = next(test_iter)
            content = content.to(device)
            style = style.to(device)
            with torch.no_grad():
                out = model.generate(content, style)
                #after decoder
            #return img
            content = denorm(content, device)
            style = denorm(style, device)
            out = denorm(out, device)
            res = torch.cat([content, style, out], dim=0)
            res = res.to('cpu')
            save_image(res, f'{image_dir}/{e}_epoch_{e}_iteration.png', nrow=batch_size)
        
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
        
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
