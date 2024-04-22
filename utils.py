import torch


def ls_equ(recv:torch.Tensor,f_ideal:torch.Tensor,idx,repeat_num=8,symbol_num=12,symbol_len=1024):
    # recv:B,1,L,2
    B,C,L,_ = recv.size()
    recv = recv[:,0,:,0] + 1j* recv[:,0,:,1] # B,L
    eps = 1e-3
    recv = recv.view(B,symbol_num,symbol_len)
    f_recv = torch.fft.fft(recv,dim=2) #(B,8,symbol_len)
    f_ideal_band = f_ideal.repeat(B,repeat_num,1) #(B,repeat_num,127)
    m2 = f_recv.abs().max(dim=2).values.max(dim=1).values    
    f_recv = torch.div(f_recv,m2.unsqueeze(dim=1).unsqueeze(dim=2))
    H = torch.div(f_recv[:,:repeat_num,idx],f_ideal_band+eps) #(B,repeat_num,127)
    H = H.mean(dim = 1).unsqueeze(1).repeat(1,symbol_num,1) #(B,128)-(B,1,128)-(B,
    f_recv[:,:,idx] = torch.div(f_recv[:,:,idx],H+eps)
    recv = torch.fft.ifft(f_recv,dim=2)
    recv = recv.reshape(B,1,L)
    recv = recv-recv.mean(dim=2).repeat(1,L).unsqueeze(dim=1)
    recv = torch.div(recv,recv.std(dim=2).repeat(1,L).unsqueeze(dim=1))
    recv = torch.view_as_real(recv)
    return recv#B,

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


if __name__ == '__main__':
    # Assume B, C, L, and L1 are defined
    x = torch.randn(2,1,12288,2)
    