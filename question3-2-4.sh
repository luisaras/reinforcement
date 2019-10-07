ALPHA=0.5
GAMMA=0.9
EPSILON=0.1

for LAMBDA in 0 0.1 0.3 0.5 0.7 0.9 1 
do
	python2 gridworld.py --lambda $LAMBDA -q -a s -n 0.0 -r 0.04 -k 10 -l $ALPHA -d $GAMMA -e $EPSILON > deterministic$LAMBDA
done

for LAMBDA in 0 0.1 0.3 0.5 0.7 0.9 1 
do
	python2 gridworld.py --lambda $LAMBDA -q -a s -n 0.2 -r 0.04 -k 10 -l $ALPHA -d $GAMMA -e $EPSILON > stochastic$LAMBDA
done

#python2 pacman.py -p PacmanSarsaAgent -x 2000 -n 2010 -l smallGrid -a lamda=0