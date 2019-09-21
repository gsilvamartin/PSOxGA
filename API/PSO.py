import random
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#Detalhes dos parâmetros:

#NP (opcional [int]): tamanho da população
#C1 (opcional [float]): componente cognitivo
#C2 (opcional [float]): componente social
#w (Opcional [float]): peso inercial
#vMin (opcional [float]): velocidade mininal
#vMax (opcional [float]): velocidade máxima
#** ukwargs: argumentos adicionais

for i in range(1):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, logger=True, benchmark=Sphere())
    algo = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

    task.plot()