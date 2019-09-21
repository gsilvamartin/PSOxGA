import time

#Importações NiaPy
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import MutationUros, CrossoverUros
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#=======================================================================
#Detalhes dos parâmetros:

#NP (Opcional [int]): tamanho da população.
#Ts (opcional [int]): seleção de torneio.
#Mr (Opcional [int]): Taxa de mutação.
#Cr (opcional [float]): taxa de cruzamento.
#Selection (Opcional [Callable [[numpy.ndarray [Individual], int, int, Individual, mtrand.RandomState], Individual]]): Operador de seleção.
#Crossover (opcional [Callable [[numpy.ndarray [Individual], int, float, mtrand.RandomState], Individual]]): operador de crossover.
#Mutação (opcional [Callable [[numpy.ndarray [Individual], int, float, Task, mtrand.RandomState], Individual]]): Operador de mutação.
#=======================================================================

#=======================================================================
#Detalhes parâmetros StoppingTask

#D (Opcional [int]): Número de dimensões.
#optType (opcional [OptimizationType]): defina o tipo de otimização. (Minimização - Máximização)
#benchmark (Union [str, Benchmark]): problema a ser resolvido com a otimização.
#nGEN (int): número máximo de iterações / gerações de algoritmos.
#nFES (int): número máximo de avaliações de funções.
#=======================================================================

tamanhoPopulacao = 100
selecaoTorneio = 0
taxaMutacao = 0.9
taxaCruzamento = 0.45
crossover = CrossoverUros
mutation = MutationUros
dimensoes = 10
numeroAvaliacoes = 4000
tipoOtimizacao = OptimizationType.MINIMIZATION
tipoBenchmark = Sphere()
iteracoes = 5

startTime = time.time()

for i in range(iteracoes):
	task = StoppingTask(D=dimensoes, nFES=numeroAvaliacoes, optType=tipoOtimizacao, benchmark=tipoBenchmark)
	algo = GeneticAlgorithm(NP=tamanhoPopulacao, Crossover=crossover, Mutation=mutation, Cr=taxaCruzamento, Mr=taxaMutacao)
	best = algo.run(task=task)
	print('%s -> %s' % (best[0].x, best[1]))

endTime = time.time()

print("Iterações:" + str(iteracoes))
print("Tempo de execução: " + str((endTime - startTime)))
task.plot()