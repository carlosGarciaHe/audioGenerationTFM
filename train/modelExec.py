import boto3
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from ipywidgets import IntProgress

import time
start_time = time.time()

sys.path.append("../src")

from Dataset import VideoDatasetAprox1, LocalVideoDatasetAprox1, VideoDatasetAprox2, LocalVideoDatasetAprox2, VideoDatasetAprox3, VideoDatasetAprox4
from Model import ModelAprox1, ModelAprox2, ModelAprox3, ModelAprox4, ModelAprox5
from Model import ModelAprox6, ModelAprox7, ModelAprox8, ModelAprox9, ModelAprox10
from Model import ModelAprox11, ModelAprox12, ModelAprox13, ModelAprox14, ModelAprox15
from Model import ModelAprox16, ModelAprox17, ModelAprox18, ModelAprox19

dataset = LocalVideoDatasetAprox2()
BATCH_SIZE = 2
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Construimos el modelo correspondiente
model = ModelAprox14(input_size=1025, output_size=256, hidden_dim=64, n_layers=1)
# Llevamos el modelo al dispositivo que lo procesará
device = torch.device(0)
model.cuda()

# Definimos hiperparámetros
n_epochs = 100
lr=0.01

# Definimos función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_epochs=1
print("Inicio entrenamiento")
for epoch in range(1, n_epochs + 1):
    n_batch = 0
    for batch in iter(dataloader):
        input_seq = batch[0]
        target_seq = input_seq.view(-1).long()[1:]
        target_seq = torch.cat((target_seq, torch.tensor([float(target_seq[-1].numpy())])), 0).long()
        batch = batch
        target_seq = target_seq.cuda()
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        output = model(batch)
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        n_batch += BATCH_SIZE
        print('batch: {}/{}.............'.format(n_batch, len(dataset)), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()))
torch.save(model, "model17")
print("--- %s seconds ---" % (time.time() - start_time))
