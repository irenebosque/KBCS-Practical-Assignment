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
## Task 2.1: Understanding the code

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
```matlab
%% Parameters
if (par.epsilon <= 0 || par.epsilon >= 1)
    error('Random action rate out of bounds, check get_parameters/par.epsilon');
```
"If epsilon is negative or zero OR greater than 1". **So epsilon has to be greater than 0 and less than 1**
If we go to file **swingup.m**, there, in line 120 we see this:
```matlab
function par = get_parameters(par)
    % TODO: set the values
    par.epsilon = 0;        % Random action rate
```
Epsilon represents the **Random action rate** and We want our agent to explore by taking random actions at times.

---


## Task 2.2: Setting the learning parameters
Look at the get_parameters function in swingup.m and set the random action rate to 0.1, and the learning rate to 0.25.

```matlab
function par = get_parameters(par)
    % TODO: set the values
    par.epsilon = 0.1; ⬅️     % Random action rate
    par.gamma = 0.99;       % Discount rate
    par.alpha = 0.25;  ⬅️     % Learning rate
    par.pos_states = 0;     % Position discretization
    par.vel_states = 0;     % Velocity discretization
    par.actions = 0;        % Action discretization
    par.trials = 0;         % Learning trials
end
```
a) Learning is faster with higher learning rates. Why would we want to keep it low anyway?
My answer: The steps you take are bigger but this could make it impossible to find an optimal solution

Professor's: High learning rate means that new solution is favored against already collected knowledge. Since new solution may be noisy, in order to improve robustness we shall mix (average) new information with already known.

Set the action discretization to 5 actions. Set the amount of trials to 2000. Set the position discretization to 31. Set the velocity discretization to 31. 

Run assignment_verify to make sure that you didn’t make any obvious mistakes.

```matlab
function par = get_parameters(par)
    % TODO: set the values
    par.epsilon = 0.1;      % Random action rate
    par.gamma = 0.99;       % Discount rate
    par.alpha = 0.25;       % Learning rate
    par.pos_states = 31; ⬅️    % Position discretization
    par.vel_states = 31; ⬅️    % Velocity discretization
    par.actions = 5;   ⬅️     % Action discretization
    par.trials = 2000; ⬅️        % Learning trials
end
```

Error you get at this point is:
```
Output argument "Q" (and maybe others) not assigned during call to "swingup>init_Q".

Error in assignment_verify (line 35)
Q = learner.init_Q(par);
```

## Task 2.3. Initialization
The **initial values in your Q table** can be very important for the exploration behavior, and there are therefore many ways of initializing them (see S&B, Section 2.7). This is done in the init_Q function. ⚠️ Answer is in chapter 2.6

a) Pick a method and give a short argumentation for your choice.
Choose “optimistic initial values” strategy. Setting initial value +5 will encourage exploration even if a greedy method is used. Random initialization is also possible.

b) Implement your choice. The Q table should be of size N × M × O, where N is the number of position states, M is the number of velocity states, and O is the number of actions.

In file **swingup.m** in line 129:

```matlab
function Q = init_Q(par)
    % TODO: Initialize the Q table.
    Q = ones(par.pos_states, par.vel_states, par.actions) * 5;
end
```
