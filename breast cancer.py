# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:05:38 2019

Aula 48 - Curso Redes Neurais

@author: Luiz Paulo
"""


import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def derivadaSigmoide(resultadoSigmoide):
    return resultadoSigmoide * (1 - resultadoSigmoide)

base = datasets.load_breast_cancer()

entradas = base.data

valoresSaida = base.target

saidas = np.empty([569, 1], dtype=int)

for i in range(569):
    saidas[i] = valoresSaida[i]
     

pesos0 = 2 * np.random.random((30, 5)) - 1    

pesos1 = 2 * np.random.random((5, 1)) - 1                 

 
epocas = 1000000
taxaAprendizagem = 0.5

momento = 1
taxaDeAcerto = 0.0

for j in range(epocas):    
    #ENTRADA INICIAL DOS NEURONIOS
    camadaEntrada = entradas
    #RESULTADO DA ATIVAÇÃO PARA CAMADA OCULTA
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    #CAMADA OCULTA POPULADA COM VALORES PARA RESULTADO FINAL
    camadaOculta = sigmoid(somaSinapse0)
    #RESULTADO DA ATIVAÇÃO PARA CAMADA DE SAIDA
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    #CAMADA DE SAIDA POPULADA COM VALOR FINAL
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(abs(erroCamadaSaida))
    #print("Erro : ", str(mediaAbsoluta))
    
    derivadaSaida = derivadaSigmoide(camadaSaida)
    
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    
    deltaCamadaOculta = deltaSaidaXPeso * derivadaSigmoide(camadaOculta) 
    
    camadaOcultaTransposta = camadaOculta.T
    
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

print("Erro : ", str(mediaAbsoluta))

taxaDeAcerto = 100 - (mediaAbsoluta * 100)
print("Taxa de acerto : ", str(taxaDeAcerto))