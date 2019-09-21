import time

#Importações NiaPy
import random
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#=======================================================================
#Detalhes dos parâmetros:

#NP (opcional [int]): tamanho da população
#C1 (opcional [float]): componente cognitivo
#C2 (opcional [float]): componente social
#w (Opcional [float]): peso inercial
#vMin (opcional [float]): velocidade minina
#vMax (opcional [float]): velocidade máxima
#** ukwargs: argumentos adicionais
#=======================================================================

#=======================================================================
#Detalhes parâmetros StoppingTask

#D (Opcional [int]): Número de dimensões.
#optType (opcional [OptimizationType]): defina o tipo de otimização. (Minimização - Máximização)
#benchmark (Union [str, Benchmark]): problema a ser resolvido com a otimização.
#nGEN (int): número máximo de iterações / gerações de algoritmos.
#nFES (int): número máximo de avaliações de funções.
#=======================================================================

tamanhoPopulacao = 40
componenteCognitivo = 2.0
componenteSocial = 2.0
pesoInercial = 0.7
velocidadeMinima = -4 
velocidadeMaxima = 4
dimensoes = 10
numeroAvaliacoes = 1000
tipoOtimizacao = OptimizationType.MINIMIZATION
tipoBenchmark = Sphere()
iteracoes = 5

startTime = time.time()

for i in range(iteracoes):
    task = StoppingTask(D=dimensoes, nFES=numeroAvaliacoes, optType=tipoOtimizacao, logger=True, benchmark=tipoBenchmark)
    algo = ParticleSwarmAlgorithm(NP=tamanhoPopulacao, C1=componenteCognitivo, C2=componenteSocial, w=pesoInercial, vMin=velocidadeMinima, vMax=velocidadeMaxima)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

endTime = time.time()

print("Iterações:" + str(iteracoes))
print("Tempo de execução: " + str((endTime - startTime)))

task.plot()