from torch.utils.data import Dataset
from torchvision.io import read_video
from torchaudio.transforms import MuLawEncoding
import torch
import boto3
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
import pickle

TMP_VIDEO_PATH = '/tmp/video.mp4'

class VideoDatasetAprox1(Dataset):
    
    def __init__(self, bucket='tfm-historico-videos', set_selected="train"):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        if set_selected=="train":
            self.videos = [object for object in self.bucket.objects.all()][:60157]
        elif set_selected=="val":
            self.videos = [object for object in self.bucket.objects.all()][60157:80209]
        else:
            self.videos = [object for object in self.bucket.objects.all()][80209:]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Descargamos el video en /tmp
        self.bucket.download_file(video.key, TMP_VIDEO_PATH)
        # Cargamos el video en memoria
        video = read_video(TMP_VIDEO_PATH)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:, None]
        return sound_frames
    
class LocalVideoDatasetAprox1(Dataset):
    
    def __init__(self, path='/home/carlos_mds/data/', set_selected="train"):
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        if set_selected=="train":
            self.videos = os.listdir(path)[:60157]
        elif set_selected=="val":
            self.videos = os.listdir(path)[60157:80209]
        else:
            self.videos = os.listdir(path)[80209:]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        self.path = path
        
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Cargamos el video en memoria
        video = read_video(self.path+video)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:, None]
        return sound_frames


class VideoDatasetAprox2(Dataset):
    
    def __init__(self, bucket='tfm-historico-videos'):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        self.videos = [object for object in self.bucket.objects.all()]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
        pretrained_vision_model = models.googlenet(pretrained=True)
        #custom_head1 = nn.Sequential(nn.Linear(in_features=1000, out_features=1000))
        #pretrained_vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #self.vision_model = nn.Sequential(pretrained_vision_model,custom_head1)
        self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-1])
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Descargamos el video en /tmp
        self.bucket.download_file(video.key, TMP_VIDEO_PATH)
        # Cargamos el video en memoria
        video = read_video(TMP_VIDEO_PATH)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        video_frames = torch.movedim(video_frames, -1, 1).float()
        video_frames = self.vision_model(video_frames)
        #video_frames = video_frames.repeat_interleave(534, dim=0)
        #video_frames = video_frames[:160000,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:, None]
        return (sound_frames,video_frames)
    

class LocalVideoDatasetAprox2(Dataset):
    
    def __init__(self, path='/home/carlos_mds/data/', set_selected="train", get_image_embeddings=False):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        if set_selected=="train":
            self.videos = os.listdir(path)[:60157]
        elif set_selected=="val":
            self.videos = os.listdir(path)[60157:80209]
        else:
            self.videos = os.listdir(path)[80209:]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
        pretrained_vision_model = models.googlenet(pretrained=True)
        pretrained_vision_model.cuda()
        #custom_head1 = nn.Sequential(nn.Linear(in_features=1000, out_features=1000))
        #pretrained_vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #self.vision_model = nn.Sequential(pretrained_vision_model,custom_head1)
        self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-1])
        self.path = path
        self.get_image_embeddings = get_image_embeddings
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        if self.get_image_embeddings:
            if not os.path.isfile('/home/carlos_mds/images_embeddings/%d.pkl'%idx):
                video = self.videos[idx]
                video = read_video(self.path+video)
                video_frames = video[0][:150,:,:,:]
                video_frames = torch.movedim(video_frames, -1, 1).float().cuda()
                video_frames = self.vision_model(video_frames)
                with open('/home/carlos_mds/images_embeddings/%d.pkl'%idx, 'wb') as handle:
                    pickle.dump(video_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return 0
        else:
            video = self.videos[idx]
            video = read_video(self.path+video)
            with open('/home/carlos_mds/images_embeddings/%d.pkl'%idx, 'rb') as handle:
                video_frames = pickle.load(handle)
        #video_frames = video_frames.repeat_interleave(534, dim=0)
        #video_frames = video_frames[:160000,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:, None]
        return (sound_frames,video_frames)
        
        
class VideoDatasetAprox3(Dataset):
    
    def __init__(self, bucket='tfm-historico-videos'):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        self.videos = [object for object in self.bucket.objects.all()]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Descargamos el video en /tmp
        self.bucket.download_file(video.key, TMP_VIDEO_PATH)
        # Cargamos el video en memoria
        video = read_video(TMP_VIDEO_PATH)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:]
        return sound_frames
    
    
class LocalVideoDatasetAprox3(Dataset):
    
    def __init__(self, path='/home/carlos_mds/data/', set_selected="train"):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        if set_selected=="train":
            self.videos = os.listdir(path)[:60157]
        elif set_selected=="val":
            self.videos = os.listdir(path)[60157:80209]
        else:
            self.videos = os.listdir(path)[80209:]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        self.path = path
        
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Descargamos el video en /tmp
        video = read_video(self.path+video)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:]
        return sound_frames
    
    
class VideoDatasetAprox4(Dataset):
    
    def __init__(self, bucket='tfm-historico-videos'):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        self.videos = [object for object in self.bucket.objects.all()]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
        pretrained_vision_model = models.googlenet(pretrained=True)
        #custom_head1 = nn.Sequential(nn.Linear(in_features=1000, out_features=1000))
        #pretrained_vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #self.vision_model = nn.Sequential(pretrained_vision_model,custom_head1)
        self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-1])
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        # Seleccionamos el video de la lista de objetos del bucket
        video = self.videos[idx]
        # Descargamos el video en /tmp
        self.bucket.download_file(video.key, TMP_VIDEO_PATH)
        # Cargamos el video en memoria
        video = read_video(TMP_VIDEO_PATH)
        # Ya que hay videos de diferente longitud, estandarizamos a los 10 primeros segundos del video
        video_frames = video[0][:150,:,:,:]
        video_frames = torch.movedim(video_frames, -1, 1).float()
        video_frames = self.vision_model(video_frames)
        #video_frames = video_frames.repeat_interleave(534, dim=0)
        #video_frames = video_frames[:160000,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:]
        return (sound_frames,video_frames)
    
    
class LocalVideoDatasetAprox4(Dataset):
    
    def __init__(self, path='/home/carlos_mds/data/', set_selected="train", get_image_embeddings=False):
        # Configuramos el acceso al bucket seleccionado ya que el primer acceso es el más costoso en tiempo
        # por lo que conviene realizarlo en el constructor
        # Guardamos la lista de objetos al bucket asumiendo que todos los videos se encuentran en la raíz del bucket
        if set_selected=="train":
            self.videos = os.listdir(path)[:60157]
        elif set_selected=="val":
            self.videos = os.listdir(path)[60157:80209]
        else:
            self.videos = os.listdir(path)[80209:]
        # Declaramos la función que discretiza el sonido
        self.muTransform = MuLawEncoding()
        # Inicializamos el tensor con 1 para forzar el rango [-1,1]
        self.ones = torch.ones(1, 80000)
        # Inicializamos el tensor con -1 para forzar el rango [-1,1]
        self.negative_ones = torch.negative(torch.ones(1, 80000))
        
        pretrained_vision_model = models.googlenet(pretrained=True)
        #custom_head1 = nn.Sequential(nn.Linear(in_features=1000, out_features=1000))
        #pretrained_vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-2])
        #self.vision_model = nn.Sequential(pretrained_vision_model,custom_head1)
        self.vision_model = nn.Sequential(*list(pretrained_vision_model.children())[:-1])
        
        self.path = path
        self.get_image_embeddings = get_image_embeddings
    def __len__(self):
        return len(self.videos)
    
    def size(self):
        # Devolvemos un string con la memoria que ocupa el total de videos
        return str(sum([object.size for object in self.videos])/1000000000)+" GB"
    
    def __getitem__(self, idx):
        if self.get_image_embeddings:
            if not os.path.isfile('/home/carlos_mds/images_embeddings/%d.pkl'%idx):
                video = self.videos[idx]
                video = read_video(self.path+video)
                video_frames = video[0][:150,:,:,:]
                video_frames = torch.movedim(video_frames, -1, 1).float().cuda()
                video_frames = self.vision_model(video_frames)
                with open('/home/carlos_mds/images_embeddings/%d.pkl'%idx, 'wb') as handle:
                    pickle.dump(video_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return 0
        else:
            video = self.videos[idx]
            video = read_video(self.path+video)
            with open('/home/carlos_mds/images_embeddings/%d.pkl'%idx, 'rb') as handle:
                video_frames = pickle.load(handle)
        #video_frames = video_frames.repeat_interleave(534, dim=0)
        #video_frames = video_frames[:160000,:,:,:]
        # Estandarizamos el sonido a 160.000 samples
        sound_frames = video[1][0,:80000]
        # Aunque el audio está normalizado puede haber valores con valor residual que sobrepasa
        # el rango [-1, 1] por lo que forzamos a que entren dentro de dichos rangos
        sound_frames = torch.where(sound_frames>1,self.ones,sound_frames)
        sound_frames = torch.where(sound_frames<-1,self.negative_ones,sound_frames)
        # Discretizamos la señal de audio en 256 posibles valores
        sound_frames = self.muTransform.forward(sound_frames)
        # Cambiamos a tipo float el tensor ya que las RNN requieren de este tipo
        sound_frames = torch.tensor(sound_frames, dtype=torch.float)
        # Redimensionamos el tensor al formato deseado (160.000, 1)
        sound_frames = sound_frames[0,:]
        return (sound_frames,video_frames)