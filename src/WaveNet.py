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
        self.skip_convolution = nn.Conv1d(hidden_channels, 64, 1).cuda()
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
    
    
class OutputModule(nn.Module):
    def __init__(self, channels=256):
        super(OutputModule, self).__init__()
        
        self.relu = nn.ReLU()
        self.convolution1 = nn.Conv1d(64, 128, 1)
        self.convolution2 = nn.Conv1d(128, channels, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = torch.sum(x, dim=0)
        out = self.relu(out)
        out = self.convolution1(out)
        out = self.relu(out)
        out = self.convolution2(out)
        out = self.softmax(out)

        return out



class WaveNet(torch.nn.Module):
    def __init__(self, in_channels=256):
        super(WaveNet, self).__init__()
        # Aplicamos la convolución causal para redimensionar el input a la dimensionalidad interna del modelo (16 en el paper)
        self.hidden_channels = 16
        self.skip_channels = 256
        self.out_channels = 256
        self.causalInputConvolution = CausalInputConvolution(in_channels=in_channels,hidden_channels=self.hidden_channels)
        self.waveNetStack = WaveNetStack(n_layers=10,
                                         n_sequences=5,
                                         hidden_channels=self.hidden_channels,
                                         skip_channels=self.skip_channels)
        self.outputModule = OutputModule(channels=self.skip_channels)


    def forward(self, x):
        out = self.causalInputConvolution(x)
        out = self.waveNetStack(out)
        out = self.outputModule(out)
        return out