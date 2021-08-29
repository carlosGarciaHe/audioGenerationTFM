import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from random import randint
from WaveNet import WaveNet
from BatchPredictionWaveNet import BatchPredictionWaveNet
torch.manual_seed(1)

class ModelAprox1(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox1, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox2(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox2, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden1, hidden2 = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden1, hidden2) = self.rnn(x, (hidden1, hidden2))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        hidden2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden1, hidden2
    
    
class ModelAprox3(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox3, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox4(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox4, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding = nn.Linear(1024, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,randint(0,149),:,0,0][:,None,:].cuda()
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(sound, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.movedim(self.fc_image_embedding(x),0,1)
        return hidden


class ModelAprox5(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox5, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding1 = nn.Linear(1024, hidden_dim)
        self.fc_image_embedding2 = nn.Linear(1024, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,randint(0,149),:,0,0][:,None,:]
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden1, hidden2  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden1, hidden2) = self.rnn(sound, (hidden1, hidden2))
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden1 = torch.movedim(self.fc_image_embedding1(x), 0, 1)
        hidden2 = torch.movedim(self.fc_image_embedding2(x), 0, 1)
        return hidden1, hidden2
    
    
class ModelAprox6(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ModelAprox6, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #         pretrained_vision_model = models.inception_v3(pretrained=True)
        #         self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding = nn.Linear(1024, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,randint(0,149),:,0,0][:,None,:]
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(sound, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.movedim(self.fc_image_embedding(x), 0, 1)
        return hidden
    
    
class ModelAprox7(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, max_video_embeddings=10):
        super(ModelAprox7, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_video_embeddings = max_video_embeddings
        
        self.n_video_frames = 150

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding = nn.Linear(1024*max_video_embeddings, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        list_video_frames = []
        space_bt_frames = int(self.n_video_frames/self.max_video_embeddings)
        for i in range(0,self.n_video_frames,15):
            list_video_frames.append(x[1][:,i,:,0,0][:,None,:])
        list_video_frames = tuple(i for i in list_video_frames)
        video = torch.cat(list_video_frames, 2)
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(sound, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.movedim(self.fc_image_embedding(x), 0,1)
        return hidden
    
    
class ModelAprox8(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, max_video_embeddings=10):
        super(ModelAprox8, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_video_embeddings = max_video_embeddings
        
        self.n_video_frames = 150

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding1 = nn.Linear(1024*max_video_embeddings, hidden_dim)
        self.fc_image_embedding2 = nn.Linear(1024*max_video_embeddings, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        list_video_frames = []
        space_bt_frames = int(self.n_video_frames/self.max_video_embeddings)
        for i in range(0,self.n_video_frames,15):
            list_video_frames.append(x[1][:,i,:,0,0][:,None,:])
        list_video_frames = tuple(i for i in list_video_frames)
        video = torch.cat(list_video_frames, 2)
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden1, hidden2  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(sound, (hidden1, hidden2))
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden1 = torch.movedim(self.fc_image_embedding1(x), 0, 1)
        hidden2 = torch.movedim(self.fc_image_embedding2(x), 0, 1)
        return hidden1, hidden2
    
    
class ModelAprox9(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, max_video_embeddings=10):
        super(ModelAprox9, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_video_embeddings = max_video_embeddings
        
        self.n_video_frames = 150

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_image_embedding = nn.Linear(1024*max_video_embeddings, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        list_video_frames = []
        space_bt_frames = int(self.n_video_frames/self.max_video_embeddings)
        for i in range(0,self.n_video_frames,15):
            list_video_frames.append(x[1][:,i,:,0,0][:,None,:])
        list_video_frames = tuple(i for i in list_video_frames)
        video = torch.cat(list_video_frames, 2)
        
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(video)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(sound, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, x):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.movedim(self.fc_image_embedding(x), 0, 1)
        return hidden
    
    
class ModelAprox10(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="linear"):
        super(ModelAprox10, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.upsample = nn.Upsample(scale_factor=534, mode=mode)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)
        
        
        video = self.upsample(video)
        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox11(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="linear"):
        super(ModelAprox11, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.upsample = nn.Upsample(scale_factor=534, mode=mode)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden1, hidden2  = self.init_hidden(batch_size)
        
        
        video = self.upsample(video)
        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, (hidden1, hidden2))
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        hidden2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden1, hidden2
    
    
class ModelAprox12(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="linear"):
        super(ModelAprox12, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.upsample = nn.Upsample(scale_factor=534, mode=mode)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)
        
        
        video = self.upsample(video)
        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox13(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="nearest"):
        super(ModelAprox13, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.deconv1 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=5)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=2)
        self.deconv6 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)
        

        video = self.deconv1(video)
        video = self.relu(video)
        video = self.deconv2(video)
        video = self.relu(video)
        video = self.deconv3(video)
        video = self.relu(video)
        video = self.deconv4(video)
        video = self.relu(video)
        video = self.deconv5(video)
        video = self.relu(video)
        video = self.deconv6(video)
        video = self.relu(video)

        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox14(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="nearest"):
        super(ModelAprox14, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        #pretrained_vision_model = models.inception_v3(pretrained=True)
        #self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.deconv1 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=5)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=2)
        self.deconv6 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden1, hidden2  = self.init_hidden(batch_size)
        

        video = self.deconv1(video)
        video = self.relu(video)
        video = self.deconv2(video)
        video = self.relu(video)
        video = self.deconv3(video)
        video = self.relu(video)
        video = self.deconv4(video)
        video = self.relu(video)
        video = self.deconv5(video)
        video = self.relu(video)
        video = self.deconv6(video)
        video = self.relu(video)

        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, (hidden1, hidden2))
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        hidden2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden1, hidden2
    
    
class ModelAprox15(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, mode="nearest"):
        super(ModelAprox15, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        pretrained_vision_model = models.inception_v3(pretrained=True)
        self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #print(self.vision_model.eval())
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Softmax Layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.deconv1 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=5)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=2)
        self.deconv6 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)
        

        video = self.deconv1(video)
        video = self.relu(video)
        video = self.deconv2(video)
        video = self.relu(video)
        video = self.deconv3(video)
        video = self.relu(video)
        video = self.deconv4(video)
        video = self.relu(video)
        video = self.deconv5(video)
        video = self.relu(video)
        video = self.deconv6(video)
        video = self.relu(video)

        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(input_seq, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden
    
    
class ModelAprox16(nn.Module):
    
    def __init__(self, in_channels):
        super(ModelAprox16, self).__init__()

        self.WaveNet = WaveNet(in_channels)   
        
    def forward(self, x):
        
        sound = x
        batch_size = sound.size(0)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.WaveNet(sound)

        return out
    
    
class ModelAprox17(nn.Module):
    
    def __init__(self, in_channels, mode="linear"):
        super(ModelAprox17, self).__init__()

        self.WaveNet = WaveNet(in_channels)
        self.upsample = nn.Upsample(scale_factor=534, mode=mode)
        
    def forward(self, x):
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)
        
        video = self.upsample(video)
        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)
        
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.WaveNet(input_seq)

        return out
    
    
class ModelAprox18(nn.Module):
    
    def __init__(self, in_channels, mode="nearest"):
        super(ModelAprox18, self).__init__()

        self.WaveNet = WaveNet(in_channels)
        self.upsample = nn.Upsample(scale_factor=534, mode=mode)
        
        self.deconv1 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=5)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=4)
        self.deconv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=2)
        self.deconv6 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=20, stride=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        sound = x[0].cuda()
        video = x[1][:,:,:,0,0]
        video = torch.movedim(video, -1, 1)
        batch_size = sound.size(0)
        
        video = self.deconv1(video)
        video = self.relu(video)
        video = self.deconv2(video)
        video = self.relu(video)
        video = self.deconv3(video)
        video = self.relu(video)
        video = self.deconv4(video)
        video = self.relu(video)
        video = self.deconv5(video)
        video = self.relu(video)
        video = self.deconv6(video)
        video = self.relu(video)

        video = video[:,:,:sound.shape[1]]
        video = torch.movedim(video, -1, 1)
        
        input_seq = torch.cat((video, sound), -1)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.WaveNet(input_seq)

        return out
    
    
class ModelAprox19(nn.Module):
    
    def __init__(self, in_channels, boost):
        super(ModelAprox19, self).__init__()

        self.BatchPredictionWaveNet = BatchPredictionWaveNet(in_channels, boost)   
        
    def forward(self, x):
        
        sound = x
        batch_size = sound.size(0)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.BatchPredictionWaveNet(sound)

        return out