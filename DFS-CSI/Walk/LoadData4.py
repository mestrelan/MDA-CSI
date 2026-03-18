import os
from nexcsi import decoder
import matplotlib.pyplot as plt

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset


def normalize_data(data):
    # Converter a lista de arrays em um único array se necessário
    if isinstance(data, list):
        data = np.array(data)
    
    # Normalizar os dados
    data_normalized = (data - np.mean(data)) / np.std(data)
    
    return data_normalized

def magnitude(lista_complexa):
    # Converter a lista em um array numpy se não for já
    if isinstance(lista_complexa, list):
        lista_complexa = np.array(lista_complexa)
    
    # Calcular as amplitudes dos números complexos
    amplitudes = np.abs(lista_complexa)
    amplitudes_tensor = torch.from_numpy(amplitudes).float()
    #amplitudes_tensor = torch.tensor(amplitudes, dtype=torch.float32)
    return amplitudes_tensor

def magnitude_fase(lista_de_ncomplexos, batch_size=100):
    n = len(lista_de_ncomplexos)
    complex_representation = []
    
    for i in range(0, n, batch_size):
        batch = lista_de_ncomplexos[i:i+batch_size]
        magnitudes = np.abs(batch)
        phases = np.angle(batch)
        #print('complex')
        #print(magnitudes)
        #print(phases)
        #print('tensor')
        magnitudes_tensor = torch.tensor(magnitudes, dtype=torch.float32)
        #print(magnitudes_tensor)
        phases_tensor = torch.tensor(phases, dtype=torch.float32)
        #print(phases_tensor)
        complex_representation.append(torch.stack((magnitudes_tensor, phases_tensor), dim=-1))
        #print(torch.cat(complex_representation, dim=0))
    return torch.cat(complex_representation, dim=0)

def path_to_csi(lista_dos_path):
    device = "raspberrypi" # nexus5, nexus6p, rtac86u
    #device = "rtac86u"

    lista_csi=[]
    for path in lista_dos_path:
        samples = decoder(device).read_pcap(path)
        #csi_data = samples.get_pd_csi()
        #csi = decoder(device).unpack(samples['csi'])
        csi = decoder(device).unpack(samples['csi'], zero_nulls=True, zero_pilots=True) #To zero the values of Null and Pilot subcarriers:
        csi=normalize_data(csi)
        #print(csi.shape)
        #print(len(csi))
        
        #csi=magnitude_fase(csi)
        #print('csi')
        #print(csi.shape)
        #print(csi[:,0,1])
        
        #csi = torch.from_numpy(csi).float() #usar quando nao estiver usando magnitude_fase
        csi=magnitude(csi)
        csi=csi[:,6:54]#tira as portadoras
        #print('csi')
        #print(csi.shape)
        #print(csi)

        #csi = csi.reshape((1, 2, 2000, 256)) #nao usar
        #csi = csi.permute(2, 0, 1) #so para magnitude e fase
        csi=csi.unsqueeze(0)

        #print('shape')
        #print(csi)
        #print(csi[0,0,:,0])
        #print(csi[0,1,:,0])
        #print(torch.var(csi[0,0,:,0]))
        #for i in range(0,25):
        #    print(i)
        #    plot_tensor(csi, series_idx=0, channel_idx=i)
        #plot_mean_variance_per_channel(csi, series_idx=0)
        lista_csi.append(csi)
    return lista_csi

def plot_tensor(tensor, series_idx=0, channel_idx=0):
    """
    Função para plotar os valores ao longo da dimensão 2000.
    
    Args:
        tensor (torch.Tensor): Tensor de dimensão (1, 2, 2000, 256).
        series_idx (int): Índice da série temporal (0 ou 1).
        channel_idx (int): Índice do canal a ser plotado (0 a 255).
    """
    # Verificar se o tensor tem o formato esperado
    if tensor.shape != (1, 2, 2000, 256):
        raise ValueError("O tensor precisa ter a dimensão (1, 2, 2000, 256).")
    
    # Selecionar os dados da série e do canal especificados
    data = tensor[0, series_idx, :, channel_idx].numpy()
    
    # Plotar os valores ao longo da dimensão 2000
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(f'Série {series_idx + 1}, Canal {channel_idx + 1}')
    plt.xlabel('Posição ao longo da dimensão 2000')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()

def plot_mean_variance_per_channel(tensor, series_idx=0):
    """
    Função para calcular e plotar a média e variância das amostras ao longo da dimensão 2000
    para cada canal.

    Args:
        tensor (torch.Tensor): Tensor de dimensão (1, 2, 2000, 256).
        series_idx (int): Índice da série temporal (0 ou 1).
    """
    # Verificar se o tensor tem o formato esperado
    if tensor.shape != (1, 2, 2000, 256):
        raise ValueError("O tensor precisa ter a dimensão (1, 2, 2000, 256).")
    
    # Selecionar os dados da série especificada
    data = tensor[0, series_idx, :, :]  # Shape (2000, 256)

    # Calcular a média e a variância ao longo da dimensão 2000 (dimensão 0)
    mean_per_channel = torch.mean(data, dim=0).numpy()  # Shape (256,)
    variance_per_channel = torch.var(data, dim=0, unbiased=False).numpy()  # Shape (256,)

    # Plotar a média
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(mean_per_channel)
    plt.title(f'Média por canal - Série {series_idx + 1}')
    plt.xlabel('Canal')
    plt.ylabel('Média')
    plt.grid(True)
    
    # Plotar a variância
    plt.subplot(2, 1, 2)
    plt.plot(variance_per_channel)
    plt.title(f'Variância por canal - Série {series_idx + 1}')
    plt.xlabel('Canal')
    plt.ylabel('Variância')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_csi_por_pessoa(pessoas):
    pcaps_sala_vazia = []
    pcaps_presenca = []
    pcaps_sentada = []
    pcaps_pe = []
    pcaps_sentar_levantar=[]
    pcaps_deitar_levantar=[]
    pcaps_deitado = []
    pcaps_andando=[]
    pcaps_correndo=[]
    pcaps_varrendo=[]
    for candidato in pessoas:
        for coleta in os.listdir(candidato):
            pcap_path = os.path.join(candidato,coleta)
            #print(coleta)
            #print(coleta[0:2])
            if coleta[0:2]=='0_':
                pcaps_sala_vazia.append(pcap_path)
            else:
                pcaps_presenca.append(pcap_path)

            if coleta[0:2]=='1_':
                pcaps_sentada.append(pcap_path)
            if coleta[0:2]=='2_':
                pcaps_sentada.append(pcap_path)
            if coleta[0:2]=='3_':
                pcaps_sentar_levantar.append(pcap_path)
            if coleta[0:2]=='4_':
                pcaps_sentada.append(pcap_path)
            if coleta[0:2]=='5_':
                pcaps_sentada.append(pcap_path)
            if coleta[0:2]=='6_':
                pcaps_pe.append(pcap_path)
            if coleta[0:2]=='7_':
                pcaps_pe.append(pcap_path)
            if coleta[0:2]=='8_':
                pcaps_pe.append(pcap_path)
            if coleta[0:2]=='9_':
                pcaps_pe.append(pcap_path)
            if coleta[0:2]=='10':
                pcaps_deitado.append(pcap_path)
            if coleta[0:2]=='11':
                pcaps_deitado.append(pcap_path)
            if coleta[0:2]=='12':
                pcaps_deitado.append(pcap_path)
            if coleta[0:2]=='13':
                pcaps_deitado.append(pcap_path)
            if coleta[0:2]=='14':
                pcaps_deitar_levantar.append(pcap_path)
            if coleta[0:2]=='15':
                pcaps_andando.append(pcap_path)  
            if coleta[0:2]=='16':
                pcaps_correndo.append(pcap_path)
            if coleta[0:2]=='17':
                pcaps_varrendo.append(pcap_path)    
    pcaps= pcaps_sala_vazia, pcaps_presenca, pcaps_sentada, pcaps_pe, pcaps_sentar_levantar, pcaps_deitar_levantar, pcaps_deitado, pcaps_andando, pcaps_correndo, pcaps_varrendo            
    
    return pcaps

def balancear_listas(lista1, lista2):
    # Calcula o tamanho mínimo entre as duas listas
    tamanho_minimo = min(len(lista1), len(lista2))
    
    # Ajusta ambas as listas para terem o mesmo tamanho
    lista1 = lista1[:tamanho_minimo]
    lista2 = lista2[:tamanho_minimo]
    
    return lista1, lista2


def pcaps2csi(pcaps):
    pcaps_sala_vazia, pcaps_presenca, pcaps_sentada, pcaps_pe, pcaps_sentar_levantar, pcaps_deitar_levantar, pcaps_deitado, pcaps_andando, pcaps_correndo, pcaps_varrendo = pcaps

    #csi_movimento = path_to_csi(pcaps_andando) + path_to_csi(pcaps_correndo) + path_to_csi(pcaps_varrendo) 
    #csi_parada = path_to_csi(pcaps_sentada) + path_to_csi(pcaps_pe) + path_to_csi(pcaps_deitado)
    
    #csi_presenca=path_to_csi(pcaps_presenca)
    #csi_sala_vazia=path_to_csi(pcaps_sala_vazia)

    #csi_andando=path_to_csi(pcaps_andando)
    #csi_correndo=path_to_csi(pcaps_correndo)
    
    #csi_sentada=path_to_csi(pcaps_sentada)
    #csi_deitado=path_to_csi(pcaps_deitado)
    
    csi_presenca=path_to_csi(pcaps_presenca)
    csi_andando=path_to_csi(pcaps_andando)

    #embaralha a lista para variar a ordem entre os candidados e garantir que esteja pegando atividades diferentes de candidados diferentes
    random.shuffle(csi_presenca)
    random.shuffle(csi_andando)

    #balanceando
    #csi_parada = csi_parada[:len(csi_movimento)]
    csi_presenca, csi_andando = balancear_listas(csi_presenca, csi_andando)
    
    ###### 
    print('tamanho da lista csi_presenca')
    print(len(csi_presenca))
    print('tamanho da lista csi_andando')
    print(len(csi_andando))

    # Concatenando as duas listas em uma única lista de dados
    csi_total = csi_presenca + csi_andando
    # Criando rótulos para os dados (1 para dados com pessoa, 0 para dados sem pessoa)
    labels = [1] * len(csi_presenca) + [0] * len(csi_andando)
    #print(labels)

    return csi_total, labels


def dividir_pessoas(diretorio):
    # Embaralha a lista para garantir aleatoriedade
    random.shuffle(diretorio)
    
    # Calcula os tamanhos das sublistas
    tamanho_total = len(diretorio)
    tamanho_treino = int(tamanho_total * 0.7)
    tamanho_validacao = int(tamanho_total * 0.2)
    
    # Divide a lista nas proporções desejadas
    treino = diretorio[:tamanho_treino]
    validacao = diretorio[tamanho_treino:tamanho_treino + tamanho_validacao]
    teste = diretorio[tamanho_treino + tamanho_validacao:]
    
    return treino, validacao, teste

def cortar_caminho(lista_paths):
    return [os.path.basename(path) for path in lista_paths]

if __name__=="__main__":

    #samples = decoder(device).read_pcap('PC_534_10012024/CSI/scans_ds2/001/0_2023_10_30_-_12_18_26_bw_80_ch_36.pcap')

    #samples = decoder(device).read_pcap('allan/0_2018_05_05_-_02_12_31.pcap')

    diretorio_ds = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'scans_ds2')
    #diretorio_ds ='C:\\Windyzone\\CSI\\scans_ds2'

    pessoas=[]

    for candidato in os.listdir(diretorio_ds):
        path_candidato = os.path.join(diretorio_ds,candidato)
        pessoas.append(path_candidato)

    print('Quantidade de voluntarios')
    print(len(pessoas))
    
    pessoas_treino, pessoas_validacao, pessoas_teste = dividir_pessoas(pessoas)
    
    #print(pessoas_teste)
    pcaps_test=get_csi_por_pessoa(pessoas_teste)
    pcaps_val=get_csi_por_pessoa(pessoas_validacao)
    pcaps_train=get_csi_por_pessoa(pessoas_treino)

    #lista com os ids das pessoas
    pessoas_treino, pessoas_validacao, pessoas_teste = cortar_caminho(pessoas_treino),cortar_caminho(pessoas_validacao),cortar_caminho(pessoas_teste)
    
    print('Dados de teste')
    csi_test, labels_test =pcaps2csi(pcaps_test)
    print('Dados de validação')
    csi_val, labels_val=pcaps2csi(pcaps_val)
    print('Dados de treinamento')
    csi_train, labels_train=pcaps2csi(pcaps_train)

    print('Input shape:')
    print(csi_train[0].shape)
    # Dividindo os dados em treinamento e teste
    #csi_train, csi_test, labels_train, labels_test = train_test_split(csi_total, labels, test_size=0.1, random_state=42,stratify=labels)

    # Dividindo os dados de treinamento em treinamento e validação
    #csi_train, csi_val, labels_train, labels_val = train_test_split(csi_train, labels_train, test_size=0.2, random_state=42,stratify=labels_train) #stratify=labels

    # Verificando os tamanhos dos conjuntos de dados
    print("Tamanho do conjunto de treinamento:", len(csi_train))
    print("Tamanho do conjunto de validação:", len(csi_val))
    print("Tamanho do conjunto de teste:", len(csi_test))


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

    #batch_size = 4#32
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=1)
    torch.save(train_loader, "train_loader.pt")
    val_loader = DataLoader(val_dataset, batch_size=20,num_workers=1)
    torch.save(val_loader, "val_loader.pt")
    test_loader = DataLoader(test_dataset, batch_size=20,num_workers=1)
    torch.save(test_loader, "test_loader.pt")  
    print('DataLoader criados')



