To run MATLAB in Linux
Open terminal and run:

```
cd /usr/local/MATLAB/R2020b/bin$
```
Once in that directory do:
```
./matlab

```
---
## Task 2.1

a) How many simulation steps are executed in a trial?

In file **swingup.m**, in line 25:

```
            % Inner loop: simulation steps
            for tt = 1:ceil(par.simtime/par.simstep)
```

Answer: par.simtime/par.simstep = 10 / 0.05 = **200** (simulation steps).
