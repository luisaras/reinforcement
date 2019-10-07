set ALPHA=0.5
set GAMMA=0.9
set EPSILON=0.1

for %%i in (1 2 3 4 5) do (

    for %%l in (0 0.1 0.3 0.5 0.7 0.9 1) do (
        py -2 gridworld.py --lambda %%l -q -a s -n 0.2 -r 0.04 -k 10 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> stochastic%%l
    )

)