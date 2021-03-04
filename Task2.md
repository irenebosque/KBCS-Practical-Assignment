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

Nowrun assignment_verify.m. This will report any basic errors in your code. 
b) What does it report?

```
Error using assignment_verify (line 14)
Random action rate out of bounds, check get_parameters/par.epsilon
```
c) Find the source of the error. Why is this value not correct? Think about what it means in terms of the learning algorithm.
```
%% Parameters
if (par.epsilon <= 0 || par.epsilon >= 1)
    error('Random action rate out of bounds, check get_parameters/par.epsilon');
```
"If epsilon is negative or zero OR greater than 1". **So epsilon has to be greater than 0 and less than 1**
If we go to file **swingup.m**, there, in line 120 we see this:
```
function par = get_parameters(par)
    % TODO: set the values
    par.epsilon = 0;        % Random action rate
```
Epsilon represents the **Random action rate** and We want our agent to explore by taking random actions at times.

