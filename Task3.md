To run MATLAB in Linux
Open terminal and run:

```
cd /usr/local/MATLAB/R2020b/bin
```
Once in that directory do:
```
./matlab

```
# Problem 3. Model-based Control (35 Points + Bonus)
Your task is to design a model-based controller for a simulated 2 link robot arm that is tracking an ellipse.
## Task 3.1. Warming Up
1. Run controller_0.m, controller_1.m, controller_2.m.
2. controller_0 is a simple tracking PD controller on the joint level, try 3 other gain settings to
improve the performance.

With the nominal gains Kp = [2000 2000] and Kd = [100 100] RMSEx = 0.1672. 

- Increasing only the Kp gains to Kp = [4000 4000] reduces that to RMSEx=0.0999. 
- In creasing only the Kd gains to Kd = [110 110] reduces that to RMSEx=0.1671. 
- Combining the two, i.e., Kp = [4000 4000] and Kd = [110 110] results in the lowest error yet RMSEx=0.0996. 
- If the gains are even higher the system quickly becomesunstable. i.e., linear control will not work very well here.
- 
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Task3.1.2_controller0.png" width="500">

3. controller_1 and controller_2 use a model-based control approach (with the perfect analytical model). Note that the PD gains are a lot lower. There are subtle differences in how the model is used. 

First is important to know the meaning of these to investigate the code:
```matlab
% tau - torques applied to joints
% th - positions of the joints (angles)
% th_d - velocities of the joints (angular velocity)
% th_dd - acceleration of the joints
% _des - desired values (reference)
% _curr - current values (measured)
% ff_ - feedforward
% fb_ - feedback
```
 <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/robot.png" width="500">

- Which control structures discussed in the lectures do they correspond to? 

  controller_1 is an open-loop **feedback** controller with an additional low-gain PD controller:
  
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/feedback.png" width="500">
  
  controller_2 is an open-loop **feedforward** controller with an additional low-gain PD controller:
  
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/feedforward.png" width="500">
  
- Switch off the feedback (PD) in both controllers. What happens? 
- Set the initial position to the desired initial position for both controllers. What happens? 
- Do the effects of switching off the feedback and setting the initial position correspond to the properties of the controllers discussed in the lecture?

  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/CodeCogsEqn.png" width="500">
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/robot_model.png" width="500">
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/feedback_part.png" width="500">
  
## 9.1.1. Open-Loop Feedforward Control
The state x(k) of the inverse model (9.2) is updated using the output of the model (9.1), see Figure 9.1. As no feedback from the process output is used, stable control is guaranteed for open-loop stable, minimum-phase systems. However, a model-plant mismatch or a disturbance d will cause a steady-state error at the process output. This error can be compensated by some kind of feedback, using, for instance, the IMC scheme presented in Section 9.1.5.
Besides the model and the controller, the control scheme contains a reference-shaping filter.
This is usually a first-order or a second-order reference model, whose task is to generate the desired dynamics and to avoid peaks in the control action for step-like references.

  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/open_loop_feedback.png" width="500">
  
## 9.1.2. Open-Loop Feedback Control
The input x(k) of the inverse model (9.2) is updated using the output of the process itself, see Figure 9.2. The controller, in fact, operates in an open loop (does not use the error between the reference and the process output), but the current output y(k)
of the process is used at each sample to update the internal statevx(k)
of the controller. Thisvcan improve the prediction accuracy and eliminate offsets. At the same time, however, the direct updating of the model state may not be desirable in the presence of noise or a significant model–plant mismatch, in which cases it can cause oscillations or instability.
Also this control scheme contains the reference-shaping filter.

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/open_loop_feedforward.png" width="500">
