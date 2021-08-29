import torch
import torch.nn as nn
import numpy as np


class DilatedCausalConvolution(nn.Module):
    def __init__(self, channels, dilation):
        super(DilatedCausalConvolution, self).__init__()
        
        self.dilated_convolution = nn.Conv1d(in_channels=channels,
                                             out_channels = channels,
                                             kernel_size=2,
                                             dilation=dilation,
                                             bias=False)
        
    def forward(self, x):  
        out = self.dilated_convolution(x)
        return out


class CausalInputConvolution(nn.Module):
    """
        Esta capa es utilizada la primera. Su objetivo es la de transformar la dimensionalidad del input como
        si de un pooling parametrizado se tratara. Es una convolución 1x1 que transforma vectores de 256 en caso
        de audio y de 1280 en caso de audio y video a 16, que es la dimensionalidad interna de WaveNet en el paper
    """
    def __init__(self, in_channels=256, hidden_channels=16):
        super(CausalInputConvolution, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.causal_convolution = nn.Conv1d(in_channels=in_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=2,
                                    padding=1,
                                    bias=False)

    def forward(self, x):
        # Recolocamos las dimensiones y pasamos a tipo float
        #x = torch.movedim(x, -1, 1).float()
        out = x.transpose(1, 2).float()
        
        
        #layers = [2 ** i for i in range(0, 10)] * 5
        #num_receptive_fields = np.sum(layers)
        #num_receptive_fields = int(num_receptive_fields)
        
        #output_size = int(out.size(2)) - num_receptive_fields
        #if output_size < 1:
        #    raise InputSizeError(int(out.size(2)), self.receptive_fields, output_size)
        #print(num_receptive_fields)
        #print(output_size)
        
        # Aplicamos la convolución causal
        out = self.causal_convolution(out)
        # Al utilizar padding=1 (ya que utilizamos un kernel de 2) sobra el último valor de los vectores
        return out[:, :, :-1]


class WaveNetBlock(nn.Module):
    def __init__(self, hidden_channels=16, skip_channels=256, dilation=1):
        super(WaveNetBlock, self).__init__()
        
        self.dilated_convolution = DilatedCausalConvolution(hidden_channels, dilation=dilation).cuda()
        self.one_dot_one_convolution = nn.Conv1d(hidden_channels, hidden_channels, 1).cuda()
        self.skip_convolution = nn.Conv1d(hidden_channels, skip_channels, 1).cuda()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.dilated_convolution(x)

        tanh_out = self.tanh(out)
        sigmoid_out = self.sigmoid(out)
        activation_out = tanh_out * sigmoid_out

        out = self.one_dot_one_convolution(activation_out)

        block_input = x[:, :, -out.size(2):]
        out = out + block_input

        skip_out = self.skip_convolution(activation_out)
        skip_out = skip_out[:, :, -70000:]

        return out, skip_out
            


class WaveNetStack(nn.Module):
    def __init__(self, n_layers=10, n_sequences=5, hidden_channels=16, skip_channels=256):
        super(WaveNetStack, self).__init__()
        
        self.n_layers = n_layers
        self.n_sequences = n_sequences
        self.sequence = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        self.list_blocks = []
        
        for sequence in range(n_sequences):
            for layer in range(n_layers):
                dilation = self.sequence[layer]
                block = WaveNetBlock(hidden_channels, skip_channels, dilation)
                self.list_blocks.append(block)
                
    def forward(self, x):
        out = x
        skip_outs = []

        for block in self.list_blocks:
            # output is the next input
            out, skip = block(out)
            skip_outs.append(skip)
        return torch.stack(skip_outs)
    
    
class OutputModuleX2(nn.Module):
    def __init__(self, channels=256):
        super(OutputModuleX2, self).__init__()
        
        self.deconvolution1 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2).cuda()
        self.deconvolution2 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=1).cuda()
        
        self.index1 = torch.IntTensor([i for i in range(0,140000) if i%2==0]).cuda()
        self.index2 = torch.IntTensor([i for i in range(0,140000) if i%2!=0]).cuda()
        
        self.mask1 = torch.IntTensor([1 if i%2==0 else 0 for i in range(0,70000)]).cuda()
        self.mask2 = torch.IntTensor([1 if i%2!=0 else 0 for i in range(0,70000)]).cuda()
        
        self.relu = nn.ReLU()
        self.convolution1 = nn.Conv1d(channels, channels, 1).cuda()
        self.convolution2 = nn.Conv1d(channels, channels, 1).cuda()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = torch.sum(x, dim=0)
        out = self.deconvolution1(out)
        out = self.relu(out)
        out = self.deconvolution2(out)
        out = self.relu(out)
        out = out[:,:,:x.shape[-1]*2]

        out1 = torch.index_select(out, -1, self.index1)
        out2 = torch.index_select(out,-1, self.index2)

        out1 = self.relu(out1)
        out1 = self.convolution1(out1)
        out1 = self.relu(out1)
        out1 = self.convolution2(out1)
        out1 = self.softmax(out1)
        
        out2 = self.relu(out2)
        out2 = self.convolution1(out2)
        out2 = self.relu(out2)
        out2 = self.convolution2(out2)
        out2 = self.softmax(out2)
        
        out1 = out1*self.mask1
        out2 = out2*self.mask2
        
        out = out1+out2

        return out
    
    
class OutputModuleX4(nn.Module):
    def __init__(self, channels=256):
        super(OutputModuleX4, self).__init__()
        
        self.deconvolution1 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2).cuda()
        self.deconvolution2 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2).cuda()
        self.deconvolution3 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=1).cuda()
        
        self.index1 = torch.IntTensor([i for i in range(0,280000) if i%4==0]).cuda()
        self.index2 = torch.IntTensor([i for i in range(0,280000) if i%4==1]).cuda()
        self.index3 = torch.IntTensor([i for i in range(0,280000) if i%4==2]).cuda()
        self.index4 = torch.IntTensor([i for i in range(0,280000) if i%4==3]).cuda()
        
        self.mask1 = torch.IntTensor([1 if i%4==0 else 0 for i in range(0,70000)]).cuda()
        self.mask2 = torch.IntTensor([1 if i%4==1 else 0 for i in range(0,70000)]).cuda()
        self.mask3 = torch.IntTensor([1 if i%4==2 else 0 for i in range(0,70000)]).cuda()
        self.mask4 = torch.IntTensor([1 if i%4==3 else 0 for i in range(0,70000)]).cuda()
        
        self.relu = nn.ReLU()
        self.convolution1 = nn.Conv1d(channels, channels, 1).cuda()
        self.convolution2 = nn.Conv1d(channels, channels, 1).cuda()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = torch.sum(x, dim=0)
        out = self.deconvolution1(out)
        out = self.relu(out)
        out = self.deconvolution2(out)
        out = self.relu(out)
        out = self.deconvolution3(out)
        out = self.relu(out)
        out = out[:,:,:x.shape[-1]*4]

        out1 = torch.index_select(out, -1, self.index1)
        out2 = torch.index_select(out,-1, self.index2)
        out3 = torch.index_select(out,-1, self.index3)
        out4 = torch.index_select(out,-1, self.index4)

        out1 = self.relu(out1)
        out1 = self.convolution1(out1)
        out1 = self.relu(out1)
        out1 = self.convolution2(out1)
        out1 = self.softmax(out1)
        
        out2 = self.relu(out2)
        out2 = self.convolution1(out2)
        out2 = self.relu(out2)
        out2 = self.convolution2(out2)
        out2 = self.softmax(out2)
        
        out3 = self.relu(out3)
        out3 = self.convolution1(out3)
        out3 = self.relu(out3)
        out3 = self.convolution2(out3)
        out3 = self.softmax(out3)
        
        out4 = self.relu(out4)
        out4 = self.convolution1(out4)
        out4 = self.relu(out4)
        out4 = self.convolution2(out4)
        out4 = self.softmax(out4)
        
        out1 = out1*self.mask1
        out2 = out2*self.mask2
        out3 = out3*self.mask3
        out4 = out4*self.mask4
        
        out = out1+out2+out3+out4

        return out
    
    
class OutputModuleX8(nn.Module):
    def __init__(self, channels=256):
        super(OutputModuleX8, self).__init__()
        
        self.deconvolution1 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.deconvolution2 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.deconvolution3 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.deconvolution4 = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=1)
        
        self.index1 = torch.IntTensor([i for i in range(0,560000) if i%8==0])
        self.index2 = torch.IntTensor([i for i in range(0,560000) if i%8==1])
        self.index3 = torch.IntTensor([i for i in range(0,560000) if i%8==2])
        self.index4 = torch.IntTensor([i for i in range(0,560000) if i%8==3])
        self.index5 = torch.IntTensor([i for i in range(0,560000) if i%8==4])
        self.index6 = torch.IntTensor([i for i in range(0,560000) if i%8==5])
        self.index7 = torch.IntTensor([i for i in range(0,560000) if i%8==6])
        self.index8 = torch.IntTensor([i for i in range(0,560000) if i%8==7])
        
        self.mask1 = torch.IntTensor([1 if i%8==0 else 0 for i in range(0,70000)])
        self.mask2 = torch.IntTensor([1 if i%8==1 else 0 for i in range(0,70000)])
        self.mask3 = torch.IntTensor([1 if i%8==2 else 0 for i in range(0,70000)])
        self.mask4 = torch.IntTensor([1 if i%8==3 else 0 for i in range(0,70000)])
        self.mask5 = torch.IntTensor([1 if i%8==4 else 0 for i in range(0,70000)])
        self.mask6 = torch.IntTensor([1 if i%8==5 else 0 for i in range(0,70000)])
        self.mask7 = torch.IntTensor([1 if i%8==6 else 0 for i in range(0,70000)])
        self.mask8 = torch.IntTensor([1 if i%8==7 else 0 for i in range(0,70000)])
        
        self.relu = nn.ReLU()
        self.convolution1 = nn.Conv1d(channels, channels, 1)
        self.convolution2 = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = torch.sum(x, dim=0)
        out = self.deconvolution1(out)
        out = self.relu(out)
        out = self.deconvolution2(out)
        out = self.relu(out)
        out = self.deconvolution3(out)
        out = self.relu(out)
        out = self.deconvolution4(out)
        out = self.relu(out)
        out = out[:,:,:x.shape[-1]*8]

        out1 = torch.index_select(out, -1, self.index1)
        out2 = torch.index_select(out,-1, self.index2)
        out3 = torch.index_select(out,-1, self.index3)
        out4 = torch.index_select(out,-1, self.index4)
        out5 = torch.index_select(out, -1, self.index5)
        out6 = torch.index_select(out,-1, self.index6)
        out7 = torch.index_select(out,-1, self.index7)
        out8 = torch.index_select(out,-1, self.index8)

        out1 = self.relu(out1)
        out1 = self.convolution1(out1)
        out1 = self.relu(out1)
        out1 = self.convolution2(out1)
        out1 = self.softmax(out1)
        
        out2 = self.relu(out2)
        out2 = self.convolution1(out2)
        out2 = self.relu(out2)
        out2 = self.convolution2(out2)
        out2 = self.softmax(out2)
        
        out3 = self.relu(out3)
        out3 = self.convolution1(out3)
        out3 = self.relu(out3)
        out3 = self.convolution2(out3)
        out3 = self.softmax(out3)
        
        out4 = self.relu(out4)
        out4 = self.convolution1(out4)
        out4 = self.relu(out4)
        out4 = self.convolution2(out4)
        out4 = self.softmax(out4)
        
        out5 = self.relu(out5)
        out5 = self.convolution1(out5)
        out5 = self.relu(out5)
        out5 = self.convolution2(out5)
        out5 = self.softmax(out5)
        
        out6 = self.relu(out6)
        out6 = self.convolution1(out6)
        out6 = self.relu(out6)
        out6 = self.convolution2(out6)
        out6 = self.softmax(out6)
        
        out7 = self.relu(out7)
        out7 = self.convolution1(out7)
        out7 = self.relu(out7)
        out7 = self.convolution2(out7)
        out7 = self.softmax(out7)
        
        out8 = self.relu(out8)
        out8 = self.convolution1(out8)
        out8 = self.relu(out8)
        out8 = self.convolution2(out8)
        out8 = self.softmax(out8)
        
        out1 = out1*self.mask1
        out2 = out2*self.mask2
        out3 = out3*self.mask3
        out4 = out4*self.mask4
        out5 = out5*self.mask5
        out6 = out6*self.mask6
        out7 = out7*self.mask7
        out8 = out8*self.mask8
        
        out = out1+out2+out3+out4+out5+out6+out7+out8

        return out



class BatchPredictionWaveNet(torch.nn.Module):
    def __init__(self, in_channels=256, boost=2):
        super(BatchPredictionWaveNet, self).__init__()
        # Aplicamos la convolución causal para redimensionar el input a la dimensionalidad interna del modelo (16 en el paper)
        self.hidden_channels = 16
        self.skip_channels = 256
        self.out_channels = 256
        self.causalInputConvolution = CausalInputConvolution(in_channels=in_channels,hidden_channels=self.hidden_channels)
        self.waveNetStack = WaveNetStack(n_layers=10,
                                         n_sequences=5,
                                         hidden_channels=self.hidden_channels,
                                         skip_channels=self.skip_channels)
        if boost==2:
            self.outputModule = OutputModuleX2(channels=self.skip_channels)
        elif boost==4:
            self.outputModule = OutputModuleX4(channels=self.skip_channels)
        elif boost==8:
            self.outputModule = OutputModuleX8(channels=self.skip_channels)


    def forward(self, x):
        out = self.causalInputConvolution(x)
        out = self.waveNetStack(out)
        out = self.outputModule(out)
        return out