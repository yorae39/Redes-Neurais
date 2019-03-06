# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:49:26 2019

Aula 50 - Curso Redes Neurais

@author: Luiz Paulo
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.structure.modules import SoftmaxLayer
#from pybrain.structure.modules import SigmoidLayer

"""
rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer, hiddenclass = SigmoidLayer, bias = False)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])
"""

rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

#print(base['input'])
#print(base['target'])

treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01, momentum = 0.06)

for i in range(1, 30000):
    erro = treinamento.train()
    if i % 10000 == 0:
        print("Erro: %s" % erro)
        
print(round(rede.activate([0, 0])))
print(round(rede.activate([0, 1])))  
print(round(rede.activate([1, 0]))) 
print(round(rede.activate([1, 1])))   
   
        
        