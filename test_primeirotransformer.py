import os
import time
from nexcsi import decoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# print(np.array(cs).shape)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.dense1 = torch.nn.Linear(2029968, 7) #foi preciso mudar esse numero para se ajustar ao entrada 256, antes estava 90 no shape (1, 2000, 90)
        # self.dense2 = torch.nn.Linear(100, 7)
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
        #     torch.nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # self.dense = torch.nn.Sequential(torch.nn.Linear(3 * 3 * 8, 100),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Dropout(p=0.5),
        #                                  torch.nn.Linear(100, 7))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # x = self.dense1(x.cuda())
        x = self.dense1(x)
        predict = self.softmax(x)
        return predict

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling layer to aggregate sequence outputs

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate along the sequence length dimension
        x = self.fc(x)
        #x = x.unsqueeze(dim=1)
        #x = self.conv1(x)
        #x = x.view(x.size(0), -1)

        # x = self.dense1(x.cuda())
        #x = self.dense1(x)
        #x = self.softmax(x)
        return x

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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(inputs.shape)
            #print(outputs.shape)
            #print(labels.shape)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

# Função para testar o modelo
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Colocar o modelo em modo de avaliação
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():  # Desabilitar o cálculo de gradientes
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) #para o AUC

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Calcular AUC (para problemas multiclasse)
    #auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    cm = confusion_matrix(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    #print(f'AUC: {auc:.4f}')

    # Plotar matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def load_data():
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
        #print(csi.shape)
        #print(len(csi))
        csi = csi.reshape((1, 2000, 256))
        csi = torch.from_numpy(csi).float()
        csi_sem_pessoa.append(csi)

    for path in pcaps_presenca:
        samples = decoder(device).read_pcap(path)
        #csi_data = samples.get_pd_csi()
        csi = decoder(device).unpack(samples['csi'])
        csi = csi.reshape((1, 2000, 256))
        csi = torch.from_numpy(csi).float()
        csi_com_pessoa.append(csi)

    #csi_com_pessoa = torch.cat(csi_com_pessoa, dim=0)
    #csi_sem_pessoa = torch.cat(csi_sem_pessoa, dim=0)

    #csi_com_pessoa = csi_com_pessoa[:len(csi_sem_pessoa)]

    print('tamanho da lista csi_com_pessoa')
    print(len(csi_com_pessoa))
    print('tamanho da lista csi_sem_pessoa')
    print(len(csi_sem_pessoa))

    # Concatenando as duas listas em uma única lista de dados
    csi_total = csi_com_pessoa + csi_sem_pessoa
    # Criando rótulos para os dados (1 para dados com pessoa, 0 para dados sem pessoa)
    labels = [1] * len(csi_com_pessoa) + [0] * len(csi_sem_pessoa)
    #print(labels)

    # Dividindo os dados em treinamento e teste
    csi_train, csi_test, labels_train, labels_test = train_test_split(csi_total, labels, test_size=0.1, random_state=42)

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


    #csi_train_normalized = normalize_data(csi_train)
    #csi_val_normalized = normalize_data(csi_val)
    #csi_test_normalized = normalize_data(csi_test)
    #print('conjuntos normalizados criados')

    # Converta-os em tensores PyTorch
    csi_train_tensor= torch.cat(csi_train, dim=0)
    csi_val_tensor= torch.cat(csi_val, dim=0)
    csi_test_tensor= torch.cat(csi_test, dim=0)

    labels_train_tensor = torch.tensor(labels_train)
    labels_val_tensor = torch.tensor(labels_val)
    labels_test_tensor = torch.tensor(labels_test)
    print('tensores criados')

    # Crie conjuntos de dados PyTorch usando TensorDataset
    train_dataset = TensorDataset(csi_train_tensor, labels_train_tensor)
    val_dataset = TensorDataset(csi_val_tensor, labels_val_tensor)
    test_dataset = TensorDataset(csi_test_tensor, labels_test_tensor)

    batch_size = 4#32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=1)
    print('DataLoader criados')
    return train_loader,val_loader,test_loader


def c_main(loader):
    model = CNN()
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.NLLLoss()
    #cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 20
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("Epoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        print("\n")
        steps = len(loader)
        for batch in tqdm(loader):
            X_train, Y_train = batch
            #Y_train = Y_train.unsqueeze(dim=1)
            #Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            loss = criterion(outputs, Y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            tr_acc += running_correct.item()
            total_num += len(batch[0])
            # running_correct += torch.sum(pred == Y_train.data)
        print("\n Loss is:", format(running_loss / steps), "Train Accuracy is", tr_acc/total_num)

if __name__=="__main__":
    try:
        inicio_temptotal=time.time()
        #c_main(loader)

        train_loader,val_loader,test_loader=load_data()
        
        #Criacao do modelo
        #model = CNN()
        
        #Transformer
        # Parâmetros do modelo
        input_size = 256  # Tamanho da entrada (número de features)
        output_size = 2  # Número de classes (posições)
        num_heads = 8  # Número de cabeças de atenção
        num_layers = 6  # Número de camadas do Transformer

        # Criar uma instância do modelo
        model = TransformerModel(input_size, output_size, num_heads, num_layers)
        print('modelo criado')

        if torch.cuda.is_available():
            model = model.cuda()
        #criterion = torch.nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
        #cost = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        inicio_temptrain=time.time()
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        fim_temptrain=time.time()
        #c_main(train_loader)
        inicio_temptest=time.time()
        test_model(model, test_loader)
        fim_temptest=time.time()
        fim_temptotal = time.time()


        print('Tempo total')
        temptotal = fim_temptotal - inicio_temptotal
        print(temptotal)
        print('Tempo de treinamento')
        temptrain = fim_temptrain - inicio_temptrain
        print(temptrain)
        print('Tempo de teste')
        temptest = fim_temptest - inicio_temptest
        print(temptest)

        print('acabou')
    except KeyboardInterrupt:
        print("error")
