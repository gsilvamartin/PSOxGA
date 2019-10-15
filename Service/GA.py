# Importações do sistema
import time

# Importações NiaPy
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import MutationUros, CrossoverUros
from NiaPy.task.task import StoppingTask, OptimizationType

# =======================================================================
# Detalhes dos parâmetros:

# NP (Opcional [int]): tamanho da população.
# Ts (opcional [int]): seleção de torneio.
# Mr (Opcional [int]): Taxa de mutação.
# Cr (opcional [float]): taxa de cruzamento.
# Selection (Opcional [Callable [[numpy.ndarray [Individual], int, int, Individual, mtrand.RandomState], Individual]]): Operador de seleção.
# Crossover (opcional [Callable [[numpy.ndarray [Individual], int, float, mtrand.RandomState], Individual]]): operador de crossover.
# Mutação (opcional [Callable [[numpy.ndarray [Individual], int, float, Task, mtrand.RandomState], Individual]]): Operador de mutação.
# =======================================================================

# =======================================================================
# Detalhes parâmetros StoppingTask

# D (Opcional [int]): Número de dimensões.
# optType (opcional [OptimizationType]): defina o tipo de otimização. (Minimização - Máximização)
# benchmark (Union [str, Benchmark]): problema a ser resolvido com a otimização.
# nGEN (int): número máximo de iterações / gerações de algoritmos.
# nFES (int): número máximo de avaliações de funções.
# =======================================================================

tamanhoPopulacao = 100
selecaoTorneio = 5
taxaMutacao = 0.9
taxaCruzamento = 0.45
crossover = CrossoverUros
mutation = MutationUros
dimensoes = 10
numeroAvaliacoes = 10000
tipoOtimizacao = OptimizationType.MINIMIZATION
iteracoes = 10000

def executeGA(typeBenchmark):
    task = StoppingTask(
        D=dimensoes,
        nFES=numeroAvaliacoes,
        optType=tipoOtimizacao,
		logger=True,
        benchmark=typeBenchmark,
    )

    algo = GeneticAlgorithm(
        NP=tamanhoPopulacao,
        Crossover=crossover,
        Mutation=mutation,
        Cr=taxaCruzamento,
        Mr=taxaMutacao,
    )

    best = algo.run(task=task)

    return [task, best[1], best[0]]

