import os
from nexcsi import decoder
from sklearn.model_selection import train_test_split

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

def normalize_data(data):
    # Converter a lista de arrays em um único array se necessário
    if isinstance(data, list):
        data = np.array(data)
    
    # Normalizar os dados
    data_normalized = (data - np.mean(data)) / np.std(data)
    
    return data_normalized


def magnitude_fase(lista_de_ncomplexos, batch_size=100):
    n = len(lista_de_ncomplexos)
    complex_representation = []
    
    for i in range(0, n, batch_size):
        batch = lista_de_ncomplexos[i:i+batch_size]
        magnitudes = np.abs(batch)
        phases = np.angle(batch)
        
        magnitudes_tensor = torch.tensor(magnitudes, dtype=torch.float32)
        phases_tensor = torch.tensor(phases, dtype=torch.float32)
        
        complex_representation.append(torch.stack((magnitudes_tensor, phases_tensor), dim=-1))
    
    return torch.cat(complex_representation, dim=0)

def check_data_loader(loader):
    for inputs, labels in loader:
        # Imprimir os primeiros exemplos de entrada e rótulos
        print("Exemplos de Entrada:")
        print(inputs[:5])  # Imprimir os primeiros 5 exemplos de entrada
        print("Rótulos Correspondentes:")
        print(labels[:5])  
        break  

#nao esta funcionando bem
def visualize_data(loader):
    for inputs, labels in loader:
        # Converter rótulos para cores usando um mapeamento de cores (colormap)
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
        colors = cmap(norm(labels))

        # Plotar os dados de entrada em relação aos rótulos
        plt.figure(figsize=(10, 6))
        plt.scatter(inputs[:, 0], inputs[:, 1], c=colors)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Rótulos')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Visualização dos Dados de Entrada em Relação aos Rótulos')
        plt.show()
        break 

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Avaliar o modelo no conjunto de validação
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc


device = "raspberrypi" # nexus5, nexus6p, rtac86u
#device = "rtac86u"

#samples = decoder(device).read_pcap('PC_534_10012024/CSI/scans_ds2/001/0_2023_10_30_-_12_18_26_bw_80_ch_36.pcap')

#samples = decoder(device).read_pcap('allan/0_2018_05_05_-_02_12_31.pcap')

diretorio_ds = '/home/monitora/Documentos/CSI/PC_534_10012024/CSI/scans_ds2'#/001'

pcaps_sala_vazia = []
pcaps_presenca = []

csi_sem_pessoa = []
csi_com_pessoa = []


for candidato in os.listdir(diretorio_ds):
	path_candidato = os.path.join(diretorio_ds,candidato)
	for coleta in os.listdir(path_candidato):
		pcap_path = os.path.join(path_candidato,coleta)
		if coleta[0]=='0':
			pcaps_sala_vazia.append(pcap_path)
		else:
			pcaps_presenca.append(pcap_path)


for path in pcaps_sala_vazia:
	samples = decoder(device).read_pcap(path)
	#csi_data = samples.get_pd_csi()
	#csi = decoder(device).unpack(samples['csi'])
	csi = decoder(device).unpack(samples['csi'], zero_nulls=True, zero_pilots=True) #To zero the values of Null and Pilot subcarriers:
	csi_sem_pessoa.append(csi)

for path in pcaps_presenca:
	samples = decoder(device).read_pcap(path)
	#csi_data = samples.get_pd_csi()
	csi = decoder(device).unpack(samples['csi'])
	csi_com_pessoa.append(csi)

print('tamanho da lista csi_com_pessoa')
print(len(csi_com_pessoa))
print('tamanho da lista csi_sem_pessoa')
print(len(csi_sem_pessoa))

# Concatenando as duas listas em uma única lista de dados
csi_total = csi_com_pessoa + csi_sem_pessoa
# Criando rótulos para os dados (1 para dados com pessoa, 0 para dados sem pessoa)
labels = [1] * len(csi_com_pessoa) + [0] * len(csi_sem_pessoa)

# Dividindo os dados em treinamento e teste
csi_train, csi_test, labels_train, labels_test = train_test_split(csi_total, labels, test_size=0.2, random_state=42)

# Dividindo os dados de treinamento em treinamento e validação
csi_train, csi_val, labels_train, labels_val = train_test_split(csi_train, labels_train, test_size=0.2, random_state=42)

# Verificando os tamanhos dos conjuntos de dados
print("Tamanho do conjunto de treinamento:", len(csi_train))
print("Tamanho do conjunto de validação:", len(csi_val))
print("Tamanho do conjunto de teste:", len(csi_test))

pcaps_sala_vazia = []
pcaps_presenca = []

csi_sem_pessoa = []
csi_com_pessoa = []
csi_total =[]


csi_train_normalized = normalize_data(csi_train)
csi_val_normalized = normalize_data(csi_val)
csi_test_normalized = normalize_data(csi_test)
print('conjuntos normalizados criados')

# Converta-os em tensores PyTorch
csi_train_tensor = torch.tensor(csi_train_normalized, dtype=torch.float32)
csi_val_tensor = torch.tensor(csi_val_normalized, dtype=torch.float32)
csi_test_tensor = torch.tensor(csi_test_normalized, dtype=torch.float32)


#csi_train_tensor=magnitude_fase(csi_train_normalized)
#csi_val_tensor=magnitude_fase(csi_val_normalized)
#csi_test_tensor=magnitude_fase(csi_test_normalized)
print('tensores criados')

# Crie os rótulos para os conjuntos de dados (1 para dados com pessoa, 0 para dados sem pessoa)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
labels_val_tensor = torch.tensor(labels_val, dtype=torch.long)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

# Crie conjuntos de dados PyTorch usando TensorDataset
train_dataset = TensorDataset(csi_train_tensor, labels_train_tensor)
val_dataset = TensorDataset(csi_val_tensor, labels_val_tensor)
test_dataset = TensorDataset(csi_test_tensor, labels_test_tensor)
print('conjuntos criados')

# Defina um DataLoader para cada conjunto de dados para permitir iteração em lotes
batch_size = 32#32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print('DataLoader criados')
'''
# Verificar DataLoader de Treinamento
print("Verificando DataLoader de Treinamento:")
check_data_loader(train_loader)

# Verificar DataLoader de Validação
print("\nVerificando DataLoader de Validação:")
check_data_loader(val_loader)

# Verificar DataLoader de Teste
print("\nVerificando DataLoader de Teste:")
check_data_loader(test_loader)

# Visualizar DataLoader de Treinamento
print("Visualizando DataLoader de Treinamento:")
visualize_data(train_loader)

# Visualizar DataLoader de Validação
print("\nVisualizando DataLoader de Validação:")
visualize_data(val_loader)

# Visualizar DataLoader de Teste
print("\nVisualizando DataLoader de Teste:")
visualize_data(test_loader)
'''

# Parâmetros do modelo
input_size = 256  # Tamanho da entrada (número de features)
output_size = 2  # Número de classes (posições)
num_heads = 8  # Número de cabeças de atenção
num_layers = 6  # Número de camadas do Transformer

# Criar uma instância do modelo
model = TransformerModel(input_size, output_size, num_heads, num_layers)
print('modelo criado')
# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

print('acabou')

#
'''
print("rssi")
print(samples['rssi']) # [-75 -77 -77 ... -77 -76 -76]
print(samples['rssi'].shape)
print("fctl")
print(samples['fctl']) # [128 148 148 ... 148 148 148]
print(samples['fctl'].shape)
print("csi")
print(samples['csi'])  # [[ 19489  0  -19200  -96 -42 ...
print(samples['csi'].shape)

# samples is a Numpy Structured Array
print(samples.dtype)
print(samples.shape)

print("outros")
print("cvr")
print(samples['cvr']) 
print(samples['cvr'].shape)

print("css")
print(samples['css']) 
print(samples['css'].shape)

print("csp")
print(samples['csp']) 
print(samples['csp'].shape)


# [
#     ('ts_sec', '<u4'), ('ts_usec', '<u4'), ('saddr', '>u4'), 
#     ('daddr', '>u4'), ('sport', '>u2'), ('dport', '>u2'),
#     ('magic', '<u2'), ('rssi', 'i1'), ('fctl', 'u1'),
#     ('mac', 'u1', (6,)), ('seq', '<u2'), ('css', '<u2'),
#     ('csp', '<u2'), ('cvr', '<u2'), ('csi', '<i2', (512,))
# ]

# Accessing CSI as type complex64
csi = decoder(device).unpack(samples['csi'])

print("csi")
print(csi)
print(csi.shape)

print("csi[0]")
print(csi[0])
print(len(csi[0]))
'''