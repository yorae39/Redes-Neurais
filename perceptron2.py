# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:50:40 2018

Aula 10  - Curso Redes Neurais

@author: Luiz Paulo
"""
import numpy as np

entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma (e, p):
  # dot product / produto escalar
  return e.dot(p)

        
s = soma(entradas, pesos)        

print("O valor da soma é : ", s)


def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

r = stepFunction(s)

print("O valor da função é : ", r)