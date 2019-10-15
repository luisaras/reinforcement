set ALPHA=0.5
set GAMMA=0.9
set EPSILON=0.1

for %%i in (1 2 3 4 5) do (
    for %%s in (5 10 15 20 25 30 35 40 45 50) do (
        py -2 gridworld.py -g DiscountGrid -q -a d --plan-steps %%s -n 0.2 -r 0.04 -k 50 -l %ALPHA% -d %GAMMA% -e %EPSILON% >> dynaq%%s
    )
)