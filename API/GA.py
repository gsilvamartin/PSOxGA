from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import MutationUros, CrossoverUros
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#Detalhes dos parâmetros:

#NP (Opcional [int]): tamanho da população.
#Ts (opcional [int]): seleção de torneio.
#Mr (Opcional [int]): Taxa de mutação.
#Cr (opcional [float]): taxa de cruzamento.
#Selection (Opcional [Callable [[numpy.ndarray [Individual], int, int, Individual, mtrand.RandomState], Individual]]): Operador de seleção.
#Crossover (opcional [Callable [[numpy.ndarray [Individual], int, float, mtrand.RandomState], Individual]]): operador de crossover.
#Mutação (opcional [Callable [[numpy.ndarray [Individual], int, float, Task, mtrand.RandomState], Individual]]): Operador de mutação.

tamanhoPopulacao = 100
selecaoTorneio = 0
taxaMutacao = 0.9
taxaCruzamento = 0.45
crossover = CrossoverUros
mutation = MutationUros

for i in range(5):
	task = StoppingTask(D=10, nFES=4000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = GeneticAlgorithm(NP=100, Crossover=CrossoverUros, Mutation=MutationUros, Cr=0.45, Mr=0.9)
	best = algo.run(task=task)
	print('%s -> %s' % (best[0].x, best[1]))

task.plot()