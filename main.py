import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis

from gresnet import GRNet
from dataset import geth5datasets
from eval import evaluate, cal_train_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='train',choices=['train','test'])
    parser.add_argument('--min_width_per_group',type=int, default=2)

    return parser.parse_known_args()[0]

class modelTrain():
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
    def main(self):
        args = get_args()
        ########### begin hyper parameter ################
        # training
        batch_size = 1024
        wd=1e-5
        lr_init = 0.0001
        self.device = args.gpu
        max_epoch = 400
        # network
        min_width_per_group = args.min_width_per_group   # the channel number of each CNN group
        self.fusion_module = '' # choices: 'cs','se','fpn,'rc', ''

        # dataset
        data_ratio = [0.8,0.1,0.1]
        current_path = os.getcwd()
        dataset_path = current_path + '/data/LoRa.h5'

        # data format and shape
        in_modals = 't'
        data_len_idx = 12       # data length is data_len_idx*1024
        slice_len_idx = 6       # each slice len is 2**slice_len_idx
        downsample = 1
        train_snr= [0,32]       # train data of random SNR from Uniform[0,32] dB
        test_snr_list = [-20,-15,-10,-5,0,5,10,15,20,25,30,35]
        usels = True            # use LS equalization or not
        signal_start=0          # 
        train_path_list = ['1','2','3','4']
        test_path_list = ['5']
        wireless_channel = 'Outdoor'
        # Log settings
        self.log_freq = 5
        save_model = False      # Save the trained model or not
        log_name = f'{in_modals}_{train_snr}_mw{min_width_per_group}_L{slice_len_idx}'
        if self.fusion_module is not None:
            log_name += f'_{self.fusion_module}'
        if usels is False:
            log_name = log_name+'_nols'
        print(f'logname:{log_name}')
        ##### End hyper parameter settings #####

        ##### data shape calculate #####
        slice_len = pow(2,slice_len_idx)
        if slice_len_idx >= 16:
            slice_len = data_len_idx*1024
        slice_len = int(slice_len)
        data_len = data_len_idx*1024
        input_channels = int(data_len/slice_len/downsample)
        print(f'input_channels:{input_channels}')                    
        print(f'model_name,: GRNet, in_modals:{in_modals},min_width_per_group:{min_width_per_group},slice len idx:{slice_len_idx},data_len_idx:{data_len_idx},usels:{usels},save_model:{save_model},downsample:{downsample}')

        # prior
        f_ideal = np.load('prior/f_band.npy')
        band_width = 128
        symbol_len = 1024
        tensor1 = torch.arange(1, band_width//2, dtype=torch.int64).cuda()
        tensor2 = torch.arange(symbol_len-band_width//2, symbol_len, dtype=torch.int64).cuda()
        f_idx = torch.cat((tensor1, tensor2), dim=0)

        ##### Load and init dataset #####
        train_dataset, val_dataset, test_dataset, classnum = geth5datasets(h5file_path=dataset_path, train_path_list=train_path_list, test_path_list=test_path_list, data_ratio=data_ratio, in_modals=in_modals, slice_len=slice_len, downsample=downsample, data_len=data_len, da=['awgn'], start=signal_start)
        train_dataset.set_out_format(snr=train_snr)#
        val_dataset.set_out_format(snr=train_snr)#
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
        self.test_dataloader = {}
        for k in test_dataset:
            print(f'data num in test dataset {k}:{len(test_dataset[k])}')
            self.test_dataloader[k] = DataLoader(test_dataset[k], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
        self.num_class = classnum            

        #### init dl model and training utils ####
        model = GRNet(expand=[min_width_per_group,min_width_per_group,2*min_width_per_group,4*min_width_per_group,8*min_width_per_group],groups=input_channels,num_class=self.num_class,usels=usels,fusion=[self.fusion_module],f_ideal=f_ideal,f_idx=f_idx)
        optimizer = torch.optim.Adam(model.parameters(),lr = lr_init,weight_decay=wd)
        loss_func = torch.nn.CrossEntropyLoss()

        #### test FLOPs and model size ####
        input_test = torch.zeros(1,input_channels,slice_len,2,dtype=torch.float32)
        input_test=input_test.cuda()
        model.cuda(self.device)
        model.eval()
        flops = FlopCountAnalysis(model, input_test)
        flops = flops.total()/1.0e6
        print(f'FLOPs:{flops:.3f}M, model size:')
        get_model_size(model)
        
        ## test a trained model
        epochstart=0
        if args.task=='test':
            epochstart = self.load(args.model_path,model,optimizer)
            for k in test_dataset:
                test_acc_list = []
                test_best_top1_name_list = []
                for testSNR in test_snr_list:
                    test_dataset[k].set_out_format(snr=testSNR)#
                    test_loss, test_best_top1,test_best_top1_name,test_eval_acces = self.test_loop(self.test_dataloader[k],loss_func,model)
                    test_acc_list.append(test_best_top1)
                    test_best_top1_name_list.append(test_best_top1_name)
                    
                print(f'[finish] dateset:{k}, test snr:{test_snr_list},test acc:{test_acc_list}')
            return
        
        ### training 
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        early_stop_count = 0
        for epoch in range(epochstart,max_epoch):
            epoch_begin_time = time.time()
            train_loss, train_msg = self.train_loop(self.train_dataloader,optimizer,loss_func,model,epoch)
            train_end_time = time.time()
            val_loss, val_best_top1,val_best_top1_name,val_eval_acces = self.test_loop(self.val_dataloader,loss_func,model)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_best_top1)
            val_loss_list.append(val_loss)
            val_end_time = time.time()
            train_time = train_end_time - epoch_begin_time
            val_time = val_end_time - train_end_time

            # log and save
            if (epoch)%self.log_freq==0:
                print('---------------------')
                print(f'epoch:{epoch}, val_acc:{val_best_top1:.4f},val_loss:{val_loss}, val name:{val_best_top1_name}, train time:{train_time/60:.1f} min, val time:{val_time/60:.1f} min')
                print('train msg:',train_msg)
                print('val acces:',val_eval_acces)
            if val_best_top1 >= max(val_acc_list):
                early_stop_count = 0
                bestEpoch = epoch
                best_val_name = val_best_top1_name
                best_val_acc = val_best_top1
                checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
            else:
                early_stop_count += 1
                if early_stop_count >= 20:
                    test_acc_list = []
                    break
            
        ##### test the trained model
        for k in test_dataset:
            test_acc_list = []
            test_best_top1_name_list = []
            for testSNR in test_snr_list:
                test_dataset[k].set_out_format(snr=testSNR)
                test_loss, test_best_top1,test_best_top1_name,test_eval_acces = self.test_loop(self.test_dataloader[k],loss_func,model)
                test_acc_list.append(test_best_top1)
                test_best_top1_name_list.append(test_best_top1_name)
                print(f'[finish] snr = {testSNR},test_eval_acces:{test_eval_acces}')
                if testSNR==test_snr_list[-1]:
                    print(f'[finish] test dataset:{k}, test snr:{test_snr_list},test acc:{test_acc_list}')
                
        if save_model:
            save_path = f'model/{log_name}.pth'
            torch.save(checkpoint, save_path)
            print('The checkpoint file has been updated.')
        torch.cuda.empty_cache()
        print(f'[finish] End epoch:{epoch}, best epoch during training:{bestEpoch}, best val acc:{best_val_acc:.4f}')
        return

    def train_loop(self,dataloader,optimizer,loss_func,model,epoch):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            msg = {}
            total_loss = 0
            model.train()
            for batchidx, (X, y, snrs) in enumerate(dataloader):
                X = X.cuda(self.device)
                y = y.cuda(self.device)
                snrs = snrs.cuda(self.device)
                pred = model(X)
                loss = self.loss(loss_func,pred,y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (epoch)% self.log_freq == 0:
                    is_final = (batchidx==num_batches-1)
                    cal_train_metrics(self.fusion_module,pred,y,msg,size,is_final)
            total_loss /= size
            
            torch.cuda.empty_cache()
            return total_loss, msg

    def test_loop(self,dataloader,loss_func,model):
        torch.cuda.empty_cache()
        size = len(dataloader.dataset)
        test_loss, acc = 0, 0
        model.eval()
        best_top1, best_top1_name, eval_acces = evaluate(self.fusion_module,model,dataloader,self.device)
        with torch.no_grad():
            for batchidx, (X, y, snrs) in enumerate(dataloader):
                X = X.cuda(self.device)
                y = y.cuda(self.device)
                snrs = snrs.cuda(self.device)
                pred = model(X)
                loss = self.loss(loss_func,pred,y) 
                test_loss += loss.item()
        test_loss /= size
        acc /= size
        return test_loss, best_top1, best_top1_name, eval_acces
    
    def loss(self,loss_func,outs,labels):
        
        if self.fusion_module == 'fpn':
            loss = loss_func(outs['final'], labels)
            for i in range(4):
                loss += loss_func(outs[f'fpn{i+1}'], labels)
            return loss
        else:
            loss_acc = loss_func(outs['final'], labels)
            return loss_acc
    def load(self,check_point_path,model,optimizer):
        if os.path.isfile(check_point_path):
            print("Continue training!\n=> loading checkpoint '{}'".format(check_point_path))
            self.log_name = (check_point_path.split('/')[-1]).split('.')[0]
            checkpoint = torch.load(check_point_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']+1
            print("Loaded checkpoint, epoch start at:",epoch_start)
        else:
            print("=> no checkpoint found at '{}'".format(check_point_path))
        return epoch_start
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def worker_init_fn(worker_id):
    np.random.seed(torch.randint(0, 100000000, (1,)).item() + worker_id)
if __name__ == "__main__":
    model_train = modelTrain()
    model_train.main()

