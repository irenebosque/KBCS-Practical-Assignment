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
Run assignment_verify to find obvious mistakes.

Now the error is:
```
Output argument "s" (and maybe others) not assigned during call to "swingup>discretize_state".

Error in assignment_verify (line 52)
s = learner.discretize_state(x0, par);
```
## Task 2.4. Discretization 
In Task 3.2 (2.2)⚠️, you determined the amount of position and velocity states that your Q table can hold, and the amount of actions the agent can choose from. The state discretization is done in the **discretize_state function**. 
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/pendulum.png" width="400">

a) Implement the position discretization. The input may be outside the interval [0,2π] rad, so be sure to wrap the state around (hint: use the mod function). The resulting state must be in the range [1, par.pos_states]. This means that π rad (the “up” direction) will be in the middle of the range. See the pendulum model shown in Figure 3.

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/discretize_position.jpeg" width="700">
_Y = ceil(X)_ rounds each element of X to the **nearest integer** greater than or equal to that element.

b) Implement the velocity discretization. Even though we assume that the values will not exceed the range [−5π,5π] rads−1, they must be clipped to that range to avoid errors. The resulting state must be in the range [1,par.vel_states]. This means that zero velocity will be in the middle of the range.
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/discretize_velocity.jpeg" width="700">

In file **swingup.m** line 134:
```matlab
function s = discretize_state(x, par)
    % TODO: Discretize state. Note: s(1) should be position, s(2) velocity.

    s(1) = round((x(1)-pi)/(2*pi/par.pos_states)) + ceil(par.pos_states/2);
    s(1) = mod(s(1)-1,par.pos_states)+1;
    
    s(2) = round(x(2)/(10*pi/par.vel_states)) + ceil(par.vel_states/2);
    s(2) = min(max(s(2),1),par.vel_states);
end
```
c) What would happen if we clip the velocity range too soon, say at [−2π,2π] rads−1?
Higher velocities would not be reachable and thus pendulum might not be able to reach the top position.

Now you need to specify how the actions are turned into torque values, in the take_action function. 

d) The allowable torque is in the range [−par.maxtorque,par.maxtorque]. Distribute the actions uniformly over this range. This means that zero torque will be in the middle of the range. 
```matlab
function u = take_action(a, par)
    % TODO: Calculate the proper torque for action a. This cannot
    % TODO: exceed par.maxtorque.
    min_a = 1;
    max_a = par.actions;
    med_a = (max_a + min_a) / 2;
    u = par.maxtorque * (a - med_a)/(max_a - med_a);
end
```
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/discretize_action.jpeg" width="600">


Run assignment_verify, and look at the plots of continuous vs. discretized position. Are they what you would expect?

I get this message:
```
>> assignment_verify
Sanity checking robot_learning_control
...Parameters are within bounds
...Q value dimensionality OK
...State discretization is implemented
......Position discretization is within bounds
......Velocity discretization is within bounds
...Action execution is within bounds
```
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/task2.4-d.png" width="600">
