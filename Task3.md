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
  
**- Switch off the feedback (PD) in both controllers. What happens?**

 The model in a typical open-loop feedforward controller here is “incorported” in the reference. 
Setting Kp and Kd to 0 results in purely open-loop control. controller_1 then is still stable, the error is accumulating over time. controller_2 very quickly becomes totally unstable. 

**- Set the initial position to the desired initial position for both controllers. What happens?**
 
 If we set the initial position to the desired initial position, the error in controller_1 reduces quite a bit (but still keeps accumulating over time), controller_2 takes a little bit longer to become totally unstable. 

**- Do the effects of switching off the feedback and setting the initial position correspond to the properties of the controllers discussed in the lecture?**

 According to the lectures the open-loop feedback should be more unstable, here it is the other way around. Open-loop feedforward should be stable if the system is stable, hence the system is apparently not stable. Whether open-loop feedback can cope better with disturbances is hard to say as the open-loop feedforward is so unstable.


  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/CodeCogsEqn.png" width="500">
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/robot_model.png" width="500">
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/feedback_part.png" width="500">
  
## 9.1.1. Open-Loop Feedforward Control
The state x(k) of the inverse model (9.2) is updated using the output of the model (9.1), see Figure 9.1. As no feedback from the process output is used, stable control is guaranteed for open-loop stable, minimum-phase systems. However, a model-plant mismatch or a disturbance d will cause a steady-state error at the process output. This error can be compensated by some kind of feedback, using, for instance, the IMC scheme presented in Section 9.1.5.
Besides the model and the controller, the control scheme contains a reference-shaping filter.
This is usually a first-order or a second-order reference model, whose task is to generate the desired dynamics and to avoid peaks in the control action for step-like references.


<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/open_loop_feedforward.png" width="500">

  
## 9.1.2. Open-Loop Feedback Control
The input x(k) of the inverse model (9.2) is updated using the output of the process itself, see Figure 9.2. The controller, in fact, operates in an open loop (does not use the error between the reference and the process output), but the current output y(k)
of the process is used at each sample to update the internal statevx(k)
of the controller. Thisvcan improve the prediction accuracy and eliminate offsets. At the same time, however, the direct updating of the model state may not be desirable in the presence of noise or a significant model–plant mismatch, in which cases it can cause oscillations or instability.
Also this control scheme contains the reference-shaping filter.
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/open_loop_feedback.png" width="500">

---



# Task 3.2. Design your own Controller
The goal is to **replace** the analytical model in the feedforward part by a **data-driven model** (GP, **neural network**, fuzzy system, basis functions, etc.) or a qualitative one (naïve physics, knowledge-based, etc.). That is, you cannot make use of the physical equations and values of the analytical model. With feedback gains of Kp = [500; 500]; Kd = [50; 50]; your model needs to get a **lower RMSE than the pure PD controller** as defined in controller_0; and all that for a range of the rotational velocity **tp.w** between 70 and 80, also see controller_yours.m and controller_yours_evaluate.m For this evaluation only the feedforward model ff_yours.m can be modified (its **input parameters** are the **current joint position and velocity**, as well as the desired joint position, velocity and acceleration, not the current joint acceleration), the rest of the code (besides loading the model, variables, etc. and passing them to ff_yours) should remain functionally unchanged. For collecting data, training the model, etc. you can modify more things. You can use any toolboxes you like, however, controller_yours_evaluate.m needs to be directly run-able on a standard TUD installation (https://weblogin.tudelft.nl) after unzipping.


---

Task 3.3. Bonus Points
You can get full points for the assignment without this task. With this task you can get bonus points to make up for points you missed, the maximum grade is still a 10. You can get up to 10 points for this task plus an additional bonus if your group is among the **top 10 RMSE x scores** (lowest RMSE of all groups gets 10 points, second lowest 9 points, etc. until tenth place 1 point). The task is to make your controller **robust to additional variations in the initial joint positions** (we will evaluate a secret test set with deviations of up to ±30 deg per joint compared to the desired initial position).
