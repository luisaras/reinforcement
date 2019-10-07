# errorbars.py

import numpy as np
import matplotlib.pyplot as plt

deterministic = [
np.array([0.37665194562, 0.424321605969, 0.337613585932, 0.408796787891, 0.677103536454]), # 0.0
np.array([0.39926086737, 0.498303296662, 0.464201516311, 0.417905945946, 0.432303882151]), # 0.1
np.array([0.553814673401, 0.566984866569, 0.562218272346, 0.496900526976, 0.544083285131]), # 0.3
np.array([0.6281299066, 0.5958891526, 0.55009125209, 0.450671695385, 0.399248349421]), # 0.5
np.array([0.62230202529, 0.569115953631, 0.667710518366, 0.568010390291, 0.56754574701]), # 0.7
np.array([0.576714297195, 0.579516830027, 0.619724175097, 0.587257393289, 0.494548729882]), # 0.9
np.array([0.548818724189, 0.575167488027, 0.509744315087, 0.429611701255, 0.326181736231]) # 1.0
]

stochastic = [
np.array([0.364800970507, 0.244374215362, 0.405803047267, 0.436283204871, 0.310448878427]), # 0.0
np.array([0.422473300098, 0.399921218621, 0.454449444399, 0.395794842015, 0.383630439936]), # 0.1
np.array([0.459002481401, 0.411768411022, 0.498453325661, 0.344930635889, 0.52115535562]), # 0.3
np.array([0.489245103067, 0.457988042698, 0.597266831212, 0.377608517852, 0.406358281795]), # 0.5
np.array([0.595790748374, 0.413901674975, 0.486471248162, 0.491996342887, 0.495803185373]), # 0.7
np.array([0.377092635199, 0.447352586454, 0.441280380785, 0.412311616111, 0.552959660119]), # 0.9
np.array([0.41890777442, 0.488583912665, 0.445952838393, 0.37243180603, 0.385764500833]) # 1.0
]

def plot(samples, name): 

    avg = []
    std = []
    for i in range(len(samples)):
        avg.append(np.mean(samples[i]))
        std.append(np.std(samples[i]))
        
    tests = ["0.0", "0.1", "0.3", "0.5", "0.7", "0.9", "1.0"]
    x_pos = np.arange(len(tests))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, avg, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average returns from start state after 10 episodes')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tests)
    ax.set_title('Average returns from start state after 10 episodes for each lambda value (%s)' % name)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('lambdas_%s.png' % name)
    plt.show()

plot(deterministic, "deterministic")
plot(stochastic, "stochastic")