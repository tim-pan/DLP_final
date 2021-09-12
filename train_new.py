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
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
import adain_model
import DDAdain_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def adversarial_criterion():
    return nn.BCELoss().to(device)

def smooth_label(mode, shape):
    if mode == 'real':
        return (torch.rand(shape, device=device) / 10) + 0.9
    elif mode == 'fake':
        return torch.rand(shape, device=device) / 10
    
def main():
    '''hyperparameters'''
    batch_size = 5
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
        Ds = DDAdain_model.Discriminator_s().to(device)
        Dc = DDAdain_model.Discriminator_c().to(device)
        
    else:
        raise WrongArgumentError
    
    if reuse is not None:
        model.load_state_dict(torch.load(reuse))
        
    optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer_ds = Adam(Ds.parameters(), lr=learning_rate)
    optimizer_dc = Adam(Dc.parameters(), lr=learning_rate)

    # start training
    loss_list = []
    loss_ds_list = []
    loss_dc_list = []
    loss_gs_list = []
    loss_gc_list = []
    
    adversarial_loss = adversarial_criterion()
    for e in range(1, epoch + 1):
        num = 0
        
        #discriminator s
        running_loss_fake_d = 0
        running_loss_real_d = 0
        running_loss_for_style = 0
        running_loss_ds = 0
        running_loss_gs = 0
        running_loss_gc = 0
        running_loss_all = 0
        running_loss_adv = 0
        
        
        
#         running_loss = 0
#         running_c_loss = 0
#         running_s_loss = 0
        
#         total_loss_g = 0
#         total_loss_d = 0
        
               
        #step1##########################################
        print('step1')
        for content, style in tqdm(train_loader):
            content = content.to(device)#(bs, c, h, w)
            style = style.to(device)#(bs, c, h, w)
            content = content.detach()
            style = style.detach()
            
            num += content.size(0)
            #################################################
            # update discriminator_s
            #################################################
            
            fake = model.generate(content, style)
                
            _, _, loss_for_style = model(content, style)
            
            s_fake_prob = Ds(fake)
            s_real_prob = Ds(style)
            
            loss_fake_d = adversarial_loss(s_fake_prob, torch.zeros(s_fake_prob.shape, device=device))
            loss_real_d = adversarial_loss(s_real_prob, smooth_label('real', s_real_prob.shape))
            loss_ds = (loss_fake_d + loss_real_d + loss_for_style)/3
                     
            loss_ds_list.append(loss_ds.item())
            

            optimizer_ds.zero_grad()
            loss_ds.backward()
            optimizer_ds.step()
            
            running_loss_ds += loss_ds.item()
            running_loss_fake_d += loss_fake_d.item()
            running_loss_real_d += loss_real_d.item()
            running_loss_for_style = loss_for_style.item()
            ###############################################
            # update generator
            ###############################################
            loss_all, _, _ = model(content, style)
            
            fake_prob = Ds(fake.detach())
            
            loss_adv = adversarial_loss(fake_prob, smooth_label('real', fake_prob.shape))
        
            loss_gs = (loss_adv + loss_all)/2
            
            
            loss_gs_list.append(loss_gs.item()) 
            
            optimizer.zero_grad()
            loss_gs.backward()
            optimizer.step()
            
            running_loss_adv += loss_adv.item()
            running_loss_all += loss_all.item()
            running_loss_gs += loss_gs.item()
      
        print()
        print()
        print()
        print(f'epoch:{e} @@@@@@@@@@@@@@@@@')
        print('step1====')
        print('for discriminator_style:')
        print(f'adv loss:{((running_loss_fake_d + running_loss_real_d)/num):.4f}, running_loss_for_style:{(running_loss_for_style/num):.4f}')
        print(f'total_loss_d:{(running_loss_ds/num):.4f}')
        print()
        print('for generator:')
        print(f'adv_loss:{(running_loss_adv/num):.4f}, loss_all:{(running_loss_all/num):.4f}')
        print(f'train_loss:{(running_loss_gs/num):.4f}')
        print()
        #step2##########################################
        #discriminator_d
        running_loss_fake_d = 0
        running_loss_real_d = 0
        running_loss_for_content = 0
        running_loss_adv = 0
        running_loss_all = 0
        running_loss_dc = 0
        
        for content, style in tqdm(train_loader):
            content = content.to(device)#(bs, c, h, w)
            style = style.to(device)#(bs, c, h, w)
            content = content.detach()
            style = style.detach()
            
#             num += content.size(0)
            #################################################
            # update discriminator_c
            #################################################

            fake = model.generate(content, style)

            _, loss_for_content,_ = model(content, style)

            c_fake_prob = Dc(fake.detach())
            c_real_prob = Dc(style)

            loss_fake_d = adversarial_loss(c_fake_prob, torch.zeros(c_fake_prob.shape, device=device))
            loss_real_d = adversarial_loss(c_real_prob, smooth_label('real', c_real_prob.shape))
            loss_dc = (loss_fake_d + loss_real_d + loss_for_content)/3

            loss_dc_list.append(loss_dc.item())


            optimizer_dc.zero_grad()
            loss_dc.backward()
            optimizer_dc.step()

            running_loss_dc += loss_dc.item()
            running_loss_fake_d += loss_fake_d.item()
            running_loss_real_d += loss_real_d.item()
            running_loss_for_content = loss_for_content.item()
            ###############################################
            # update generator
            ###############################################
            
            loss_all, _, _ = model(content, style)
            fake_prob = Dc(fake)
            loss_adv = adversarial_loss(fake_prob, smooth_label('real', fake_prob.shape))

            loss_gc = (loss_adv + loss_all)/2

            loss_gc_list.append(loss_gc.item()) 

            optimizer.zero_grad()
            loss_gc.backward()
            optimizer.step()

            running_loss_adv += loss_adv.item()
            running_loss_all += loss_all.item()
            running_loss_gc += loss_gc.item()


        print(f'epoch:{e} @@@@@@@@@@@@@@@@@')
        print('step2====')
        print('for discriminator_style:')
        print(f'adv loss:{((running_loss_fake_d + running_loss_real_d)/num):.4f}, running_loss_for_content:{(running_loss_for_content/num):.4f}')
        print(f'total_loss_d:{(running_loss_dc/num):.4f}')
        print()
        print('for generator:')
        print(f'adv_loss:{(running_loss_adv/num):.4f}, loss_all:{(running_loss_all/num):.4f}')
        print(f'train_loss:{(running_loss_gc/num):.4f}')
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
        
    plt.plot(range(len(loss_gs_list)), loss_gs_list)
    plt.plot(range(len(loss_gc_list)), loss_gc_list)
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
