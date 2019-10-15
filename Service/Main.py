from matplotlib import pyplot as plt
from NiaPy.benchmarks import *
import numpy as np 
import time
import GA
import PSO

def closest(lst, K): 
    idx = (np.abs(lst - K)).argmin() 
    return lst[idx] 

benchmarkType = Salomon()

timeStartGA = time.time()
resultGA = GA.executeGA(benchmarkType)
timeEndGA = time.time()

timeStartPSO = time.time()
resultPSO = PSO.executePSO(benchmarkType)
timeEndPSO = time.time()

benchmarkTimeGA = (timeEndGA - timeStartGA)
benchmarkTimePSO = (timeEndPSO - timeStartPSO)

minorValueGA = (closest(resultGA[2].x, 0))
minorValuePSO = (closest(resultPSO[2], 0))

plt.plot(resultGA[0].n_evals, resultGA[0].x_f_vals)
plt.plot(resultPSO[0].n_evals, resultPSO[0].x_f_vals)
plt.plot(0, 0)
plt.plot(0, 0)
plt.plot(0, 0)
plt.plot(0, 0)
plt.plot(0, 0)
plt.plot(0, 0)
plt.xlabel("Número de gerações")
plt.ylabel("Fitness")
plt.title("Gráfico de convergência")

plt.legend(
    [
        "GA",
        "PSO",
        "Melhor valor(GA): " + str(round(minorValueGA, 4)),
        "Melhor valor(PSO): " + str(round(minorValuePSO, 4)),
        "Tempo GA(ms): " + str(round(benchmarkTimeGA, 4)),
        "Tempo PSO(ms): " + str(round(benchmarkTimePSO, 4)),
        "Melhor Fitness(GA): " + str(round(resultGA[1], 4)),
        "Melhor Fitness(PSO): " + str(round(resultPSO[1], 4)),
    ],
    loc="top left",
)

plt.show()

