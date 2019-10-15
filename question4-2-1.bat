set ALPHA=0.5
set GAMMA=0.9
set EPSILON=0.1

for %%i in (1 2 3 4 5) do (
    py -2 gridworld.py -q -a d --plan-steps 5 -n 0.2 -r 0.04 -k 25 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> dynaq_vs_qlearning
)

for %%i in (1 2 3 4 5) do (
    py -2 gridworld.py -q -a q -n 0.2 -r 0.04 -k 25 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> dynaq_vs_qlearning
)