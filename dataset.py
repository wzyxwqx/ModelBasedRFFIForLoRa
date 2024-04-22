import torch
from torch.utils.data import Dataset
from torch import from_numpy,stft,fft
import numpy as np
import h5py

def get_rffplora(h5file_path,group='Outdoor',train_path_list=['1','2','3','4'],test_path_list=['5'],data_shape=(1,12288,2),data_ratio=[0.8,0.1,0.1]):#(1,12288,2)
    # group in ['Indoor', 'Outdoor']
    f = h5py.File(h5file_path, "r")
    train_val_num = 0
    test_num = 0
    print(f'Days in {group}:{f[group].keys()}, train days:{train_path_list}, test days:{test_path_list}')
    for path in f[group].keys():
        if path in train_path_list:
            train_val_num += len(f[f'{group}/{path}/Y'])
        elif path in test_path_list:
            test_num += len(f[f'{group}/{path}/Y'])
    print(f'[dataset] train_val_num:{train_val_num},test_num:{test_num}')
    X1=np.zeros((train_val_num,data_shape[0],data_shape[1],data_shape[2]),dtype=np.float32)
    Y1=np.zeros((train_val_num,),dtype=np.int64)
    X2=np.zeros((test_num,data_shape[0],data_shape[1],data_shape[2]),dtype=np.float32)
    Y2=np.zeros((test_num,),dtype=np.int64)
    train_start,train_end=0,0
    test_start,test_end=0,0
    for path in f[group].keys():
        n = len(f[f'{group}/{path}/Y'])
        if path in train_path_list:
            train_end = train_end+n
            xtmp = np.array(f[f'{group}/{path}/X'])
            X1[train_start:train_end] = xtmp
            Y1[train_start:train_end] = np.array(f[f'{group}/{path}/Y']).astype(np.int64)
            train_start = train_start+n
        elif path in test_path_list:
            test_end = test_end+n
            xtmp = np.array(f[f'{group}/{path}/X'])

            X2[test_start:test_end] = xtmp
            Y2[test_start:test_end] = np.array(f[f'{group}/{path}/Y']).astype(np.int64)
            test_start = test_start+n
    train_val_indices = np.arange(train_val_num)
    np.random.shuffle(train_val_indices)
    train_num = int(data_ratio[0]*train_val_num)
    possible_val_num = int(data_ratio[1]*train_val_num)

    train_indices = train_val_indices[:train_num]
    val_indices = train_val_indices[train_num:train_num+possible_val_num]
    test_indices = train_val_indices[train_num+possible_val_num:]
    
    X_test2, Y_test2 = X2[:], Y2[:]-int(min(Y2[:]))
    if len(train_path_list)!=0:
        X_test1, Y_test1 = X1[test_indices], Y1[test_indices]-int(min(Y1[test_indices]))
        X_train, Y_train = X1[train_indices], Y1[train_indices]-int(min(Y1[train_indices]))
        X_val, Y_val = X1[val_indices], Y1[val_indices]-int(min(Y1[val_indices]))
    else:
        X_train = 0
        Y_train = 0
        X_val = 0
        Y_val = 0

    if train_path_list==test_path_list:
        X_test, Y_test = X_test1, Y_test1 
    elif len(train_path_list)==0:
        X_test, Y_test = X_test2, Y_test2 
    else:
        X_test, Y_test = (X_test1,X_test2),(Y_test1,Y_test2) 
    print('[dataset] lenth of xtest and ytest:',len(X_test),len(Y_test))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, 25

def geth5datasets(h5file_path,wireless_channel='Outdoor',data_ratio=[0.8,0.1,0.1],train_path_list=['1','2','3','4'],test_path_list=['5'],**kwargs):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, classnum = get_rffplora(h5file_path,group=wireless_channel,train_path_list=train_path_list,test_path_list=test_path_list,data_ratio=data_ratio)
    if not train_path_list:
        train_dataset, val_dataset = None, None
    else:
        train_dataset = sliceDataset(X_train,Y_train,**kwargs)
        val_dataset = sliceDataset(X_val,Y_val,**kwargs)
    if isinstance(X_test,tuple):
        test_dataset_0 = sliceDataset(X_test[0],Y_test[0],**kwargs)
        test_dataset_1 = sliceDataset(X_test[1],Y_test[1],**kwargs)
        test_dataset = {'0':test_dataset_0,'1':test_dataset_1}
        print(f'get tuple test dataset')
    else:
        test_dataset = {'0':sliceDataset(X_test,Y_test,**kwargs)}
    return train_dataset,val_dataset,test_dataset,classnum

class sliceDataset(Dataset):
    def __init__(self, X,Y, in_modals='t', slice_len=64, downsample=1, data_len=12288,da=['awgn'],start=0):
        self.X = X
        self.Y = Y
        self.in_modals = in_modals
        self.slice_len = slice_len
        self.downsample = downsample
        self.sample_num = len(Y)
        self.da=da
        self.data_len=data_len
        self.channels = self.data_len// self.slice_len //self.downsample
        self.snr=99
        self.start=start
    def update_data_len(self,data_len):
        self.data_len = int(data_len)
    def __len__(self):
       return self.sample_num
    def set_out_format(self,snr=99):
        self.snr = snr
    def __getitem__(self,index):
        wave = self.X[index]
        label = self.Y[index]
        if 'awgn' in self.da:
            wave, snr = awgn(wave,self.snr)
            wave = wave-np.mean(wave)
            wave = wave/np.std(wave)
        wave = from_numpy(wave)
        indices = np.arange(self.start, self.start + self.data_len, self.downsample)
        wave = wave[:,indices,:]
        if self.in_modals == 's':
            wave_complex = wave[:,:,0]+1j*wave[:,:,1]
            wave_s = fft.fft(wave_complex)
            wave_s_r = wave_s.real.unsqueeze(2)
            wave_s_i = wave_s.imag.unsqueeze(2)
            wave_s = torch.cat((wave_s_r,wave_s_i),dim=2)
            wave = wave_s.reshape((self.channels,self.slice_len,2))
        elif self.in_modals == 't':
            wave = wave.reshape((self.channels,self.slice_len,2))
        elif self.in_modals == 'st':
            wave_complex = wave[:,:,0]+1j*wave[:,:,1]
            wave_s = fft.fft(wave_complex)
            wave_s_r = wave_s.real.unsqueeze(2)
            wave_s_i = wave_s.imag.unsqueeze(2)
            wave_s = torch.cat((wave_s_r,wave_s_i),dim=2)
            wave_st = torch.cat((wave,wave_s),dim=0)
            wave = wave_st.reshape((self.channels*2,self.slice_len,2))
        elif self.in_modals == 'stft':
            wave_complex = wave[:,:,0]+1j*wave[:,:,1]
            wave_stft = torch.view_as_real(stft(wave_complex[0:1,:],self.slice_len,self.slice_len,self.slice_len,center=False))#(2,256,253)
            temp = (wave_stft[:, :self.slice_len//2, :]).clone().detach() 
            wave_stft[:, :self.slice_len//2, :] = wave_stft[:, self.slice_len//2:, :]
            wave_stft[:, self.slice_len//2:, :] = temp
            wave_stft = torch.squeeze(wave_stft)
            wave_stft = wave_stft.swapaxes(0,1)
            wave = wave_stft 

        return wave,label,snr
def awgn(data, SNRrange):
    if not isinstance(SNRrange,int):
        SNRdB=np.random.randint(SNRrange[-1]-SNRrange[0])+SNRrange[0]
    else:
        SNRdB= SNRrange
    SNRdB = np.array([SNRdB]).astype(np.float32)
    if SNRdB >30:
        SNRdB=np.array([35.0]).astype(np.float32)
        return data,(SNRdB-15.5)/35.0
    SNR_linear = 10**(SNRdB/10)
    P= np.mean(np.abs(data)**2)
    N0=P/SNR_linear
    n = np.sqrt(N0)*np.random.randn(*data.shape)
    n = n.astype(np.float32)
    data = data + n
    data= data/ np.std(data)
    return data, (SNRdB-15.5)/35.0



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import os
    train_path_list=[]
    print(train_path_list==[])
    print(not train_path_list)
    print(len(train_path_list))
    train_dataset,val_dataset,test_dataset,classnum = geth5datasets(os.getcwd() + '/data/LoRa.h5',train_path_list=['2'],test_path_list=['1'])
    
    print(len(train_dataset),len(val_dataset),len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    train_dataset.set_out_format(snr=99)#
    import matplotlib.pyplot as plt
    for batchidx, (X, y,snr) in enumerate(train_dataloader):
        batch_size = X.shape[0]
        print(X.shape)
        x = X.cpu().numpy()
        x = x[0,0]
        print(x.shape)
        break
