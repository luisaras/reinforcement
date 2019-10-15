import subprocess
import numpy as np
import matplotlib.pyplot as plt

from errorbars import plot

def test(command, n_times):
    results = []
    for i in range(0, n_times):
        output = str(subprocess.check_output(command))
        # Removes all text before the avarage return value
        output = output[output.find(':')+2:]
        # Removes all text after the avarage return value
        output = ''.join(c for c in output if c.isdigit())
        #strout += "\t{:<15}".format(output[:15])
        results.append(float(output))
    return np.array(results)

def test_qlearning():
    alpha=0.5
    gamma=0.9
    epsilon=0.1
    command = ["py", "-2", "gridworld.py", \
            "--agent=q", \
            "--episodes=2", \
            "--noise=0.0", \
            #"--livingReward=-0.04", \
            "--epsilon="+str(epsilon), \
            "--discount="+str(gamma), \
            "--learningRate="+str(alpha), \
            "--quiet", \
            "--text"]
    print(test(command, 5))

def test_dynaq():
    alpha=0.5
    gamma=0.9
    epsilon=0.1
    command = ["py", "-2", "gridworld.py", \
            "--agent=d", \
            "--plan-steps=50", \
            "--episodes=2", \
            "--noise=0.0", \
            #"--livingReward=-0.04", \
            "--epsilon="+str(epsilon), \
            "--discount="+str(gamma), \
            "--learningRate="+str(alpha), \
            "--quiet", \
            "--text"]
    print(test(command, 5))

def test_dynaq_vs_qlearning():
    alpha=0.5
    gamma=0.9
    epsilon=0.1
    # Q-Learning
    #py -2 gridworld.py -q -a q -n 0.0 -r 0.04 -k 25 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> dynaq_vs_qlearning
    command = ["py", "-2", "gridworld.py", \
            "--agent=q", \
            "--episodes=25", \
            "--noise=0.0", \
            "--epsilon="+str(epsilon), \
            "--discount="+str(gamma), \
            "--learningRate="+str(alpha), \
            "--quiet", \
            "--text"]
    qlearning = test(command, 5)
    # Dyna-Q
    #py -2 gridworld.py -q -a d --plan-steps 5 -n 0.0 -r 0.04 -k 25 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> dynaq_vs_qlearning
    command = ["py", "-2", "gridworld.py", \
            "--agent=d", \
            "--plan-steps=5", \
            "--episodes=25", \
            "--noise=0.0", \
            "--epsilon="+str(epsilon), \
            "--discount="+str(gamma), \
            "--learningRate="+str(alpha), \
            "--quiet", \
            "--text"] 
    dynaq = test(command, 5)
    plot([dynaq, qlearning], \
        ["Dyna-Q", "Q-Learning"], \
        "Returns from start state after 25 episodes for each algorithm", \
        "dyna-vs-ql")

def test_dynaq_plus(kappa):
    alpha=0.5
    gamma=0.9
    epsilon=0.1
    results = []
    step_values = range(5,55,5)
    for steps in step_values:
        command = ["py", "-2", "gridworld.py", \
                "--grid=DiscountGrid", \
                "--agent=d", \
                "--plan-steps="+str(steps), \
                "--kappa="+str(kappa), \
                "--episodes=50", \
                "--noise=0.0", \
                "--epsilon="+str(epsilon), \
                "--discount="+str(gamma), \
                "--learningRate="+str(alpha), \
                "--quiet", \
                "--text"]
        result = test(command, 5)
        results.append(result)
    plot(results, \
        [str(x) for x in step_values], \
        "Returns from start state after 50 episodes for each number of plan steps", \
        "dynas-discount-grid(%s)" % kappa)
    

if __name__ == '__main__':
    #test_qlearning()
    #test_dynaq()
    test_dynaq_vs_qlearning()
    test_dynaq_plus(0.0)
    test_dynaq_plus(0.1)
    test_dynaq_plus(0.5)
    exit()
    
