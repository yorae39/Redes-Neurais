# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:36:44 2019


Aula 21 a 42 - Curso Redes Neurais

@author: Luiz Paulo
"""

import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def derivadaSigmoide(resultadoSigmoide):
    return resultadoSigmoide * (1 - resultadoSigmoide)

entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    
saidas = np.array([[0],[1],[1],[0]])    

#pesos0 = np.array([[-0.424, -0.740, -0.961],
                   #[0.358, -0.577, -0.469]])
    
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])    

pesos0 = 2 * np.random.random((2, 3)) - 1    

pesos1 = 2 * np.random.random((3, 1)) - 1                 

 
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