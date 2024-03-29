

## Installation procedure
> ⚠️ The folder I download with the files it is called *KBCS-assignment-main* and not *KBCS-assignment* as indicated in the `README.md` file.

I followed this tutorial for installing anaconda on Ubuntu **20.04**: https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/

Important: Once you finish following the instructions on the tutorial, close and open a new terminal as it is indicated at the end of the tutorial.



<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_1.png" width="700">

Once it is installed, you will see a message on your terminal telling you how to activate the virtual environment:

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_2.png" width="700">
Go and activate the virtual environment: 


```bash

conda activate kbcs

```
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_3.png" width="700">

---
## Task 1.1

Once inside the virtual environment, (*Note: you are inside the virtual environment if you see (kbcs) in front of the terminal line*), run if you wish the file `generate_data.py`, it takes some time:

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/generate_data_1.png" width="700">

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/generate_data_2.png" width="700">


### Space state representation


<figure>
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/radians.png" width="200">
  <figcaption>The angle θ is defined with respect to the upright position and wrapped to the [−π,π]</figcaption>
</figure>


<figure>
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/circle_20.jpg" width="200">
  <figcaption>The provided code divides θ ∈ [−π,π] into 20 discrete bins</figcaption>
</figure>





<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/gif_task-1-2.gif" width="500">






### Task 1.1 - Create



The sequential *model_theta* can be created in one of the two following ways, like this...:

```python

model = tf.keras.Sequential(name="model_theta")
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu", name="dense_layer_128"))
model.add(tf.keras.layers.Dense(1, name="dense_layer_1"))
```
... or like this:

```python
model = tf.keras.Sequential( #irene
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", name="dense_layer_128"),
        tf.keras.layers.Dense(1, name="dense_layer_1"),
    ]
)
```
### Task 1.1 - Compile
> ⚠️ It was intentional to already include how to compile the model in `train.py`? I see that in the google notebook it is not included. Maybe this link is useful for the compile step: https://keras.io/api/optimizers/




### Task 1.1 - Train
In the pdf is written: *Make sure that we have a train-validation split of 0.2 and a batch size of 64*
> ⚠️ Maybe add something like, these 2 variables are already added in the provided `train.py` file




```python
    """TASK 1.1: TRAIN MODEL HERE"""

    model.fit(
        x=train_obs,
        y=train_theta,
        batch_size=64,
        epochs=30,
        validation_split=0.2)

    model.summary() # This will print a summary of the model in the terminal


    """TASK 1.1: END"""
```
### Task 1.1 - Run

Run the code by using the command `python train.py`:
```shell
(kbcs) irene@irene-computer:~/Dropbox/KBCS-Practical Assignment/KBCS-assignment-main$ python train.py
Loaded observation data: (12120, 28, 28, 3)
Loaded state data: (12120, 2)
Epoch 1/30
120/120 [==============================] - 1s 6ms/step - loss: 6.4024 - val_loss: 4.1779
Epoch 2/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0391 - val_loss: 4.1778
Epoch 3/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0389 - val_loss: 4.1777
.
.
.
.
.
Epoch 27/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0385 - val_loss: 4.1779
Epoch 28/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0384 - val_loss: 4.1779
Epoch 29/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0384 - val_loss: 4.1779
Epoch 30/30
120/120 [==============================] - 1s 5ms/step - loss: 5.0385 - val_loss: 4.1778


```
 
You should see in the terminal a summary of the model:
```
Model: "model_theta"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (64, 2352)                0         
_________________________________________________________________
dense_layer_128 (Dense)      (64, 128)                 301184    
_________________________________________________________________
dense_layer_1 (Dense)        (64, 1)                   129       
=================================================================
Total params: 301,313
Trainable params: 301,313
Non-trainable params: 0
_________________________________________________________________
75/75 - 0s - loss: 5.9486
Test loss: 5.9486284255981445
```




> ⚠️ In the run section maybe rewrite,  " **Manually** close the **2** generated plots to start a live simulation of your prediction model in case you run locally."


> ⚠️ To be honest, I do not perfectly understand the accuracy graph. Each bar represents one of the 20 bins in which 360 degrees are divided? Then why the X axis goes from -3pi to +3pi?

### Task 1.1 - Evaluate

Run the code multiple times and record the “average prediction error” for each run. Calculate the
mean and standard deviation of the “average prediction error” over all runs.

Task 1.1  Average predicion error: 

  - Run  1: 2.33e+00
  - Run  2: 2.00e-01
  - Run  3: 2.46e-01
  - Run  4: 5.13e-01
  - Run  5: 2.33e+00
  - Run  6: 3.23e-01
  - Run  7: 2.33e+00
  - Run  8: 2.30e-01
  - Run  9: 2.33e+00
  - Run 10: 2.57e-01

  - **Standard deviation**: 1.0544
  - **Mean**: 1.1089

Notice that the results vary across different runs. Sometimes the model learns to predict with an “average prediction error”< 1.0, while sometimes the model does not learn anything at all. Figure 2 shows two typical plots that qualitatively resemble both cases. 
How can the results be different,even though the underlying code and test dataset remained unchanged?


Analyze the “per bin average prediction errors” plot of a “successful” run (i.e. a run with a plot that resembles the left plot in Figure 2). 
- **Where does this model have the lowest accuracy**?

    At both ends, near pi and -pi
  
    <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/average_error_model_theta.png" width="500">
    <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/pendulum_model_theta.png" width="500">
  
- What could be an explanation for the loss of accuracy in that region?

    There is a discontinuity in that area. Suddenly it goes from -pi to pi



    
- In general, a separate test dataset is used to evaluate a trained model, why?

    Having 2 separate datasets helps to avoid overfitting and also it ensures that the model will generalize on data that has never seen.

## Task 1.2


### Task 1.2 - Create

The sequential *model_trig* can be created in one of the two following ways, like this...:

```python

model = tf.keras.Sequential(name="model_trig")
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu", name="dense_layer_128"))
model.add(tf.keras.layers.Dense(2, name="dense_layer_2"))
```
... or like this:

```python
model = tf.keras.Sequential( 
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", name="dense_layer_128"),
        tf.keras.layers.Dense(2, name="dense_layer_2"),
    ]
)
```


### Task 1.2 - Train
```python
    """TASK 1.2: TRAIN MODEL HERE"""

    model.fit(
        x=train_obs,
        y=train_trig,
        batch_size=64,
        epochs=30,
        validation_split=0.2)

    model.summary() # This will print a summary of the model in the terminal


    """TASK 1.1: END"""
```

> ⚠️ I manually added `model.summary()`. In the `.ipynb` it is already included

### Task 1.2 - Run

```
Model: "model_trig"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (64, 2352)                0         
_________________________________________________________________
dense_layer_128 (Dense)      (64, 128)                 301184    
_________________________________________________________________
dense_layer_2 (Dense)        (64, 2)                   258       
=================================================================
Total params: 301,442
Trainable params: 301,442
Non-trainable params: 0
_________________________________________________________________
75/75 - 0s - loss: 0.0101
Test loss: 0.010128799825906754
```
### Task 1.2 - Evaluate

Run the code multiple times and record the “average prediction error” for each run. Calculate the mean and standard deviation of the “average prediction error” over all runs.

    <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/average_error_model_trig.png" width="500">
    

  - Run  1: 2.16e-02
  - Run  2: 1.32e-02
  - Run  3: 1.05e-01
  - Run  4: 1.08e-02
  - Run  5: 6.00e-03
  - Run  6: 9.16e-03
  - Run  7: 3.01e-02
  - Run  8: 4.67e-03
  - Run  9: 2.23e-02
  - Run 10: 1.35e-02

  - **Standard deviation**: 0.0297
  - **Mean**: 0.0236

Compare the “per bin average prediction error” plot for Mtrig with the plot for Mθ. 
Why does indirectly predicting the angle improve the prediction accuracy?

| Model   | mean average prediction error |
| ------------- |:-------------:      | 
| Mθ            | 1.108            | 
| Mtrig         | 0.023           |


The results for indirectly predicting the angles with cos(θ) and sin(θ) are better in terms of the average prediction error, because there is now a continuous relationship between the angle of the pendulum and the target values of the model. Therefore, the model generalizes the results in the region around −π and
π better.


 Why is it not sufficient to predict only sin(θ) and use its inverse θ = arcsin(sin(θ)) to get an estimate of the angle?


because sin(−θ) = sin(θ).
    
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/same_sin.png" width="500">
    

    
    





## Task 1.3


### Task 1.3 - Create

The sequential *model_cnn* can be created in one of the two following ways, like this...:

```python
model = tf.keras.Sequential(name="model_cnn")
model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, name="dense_layer_2"))
```
... or like this:

```python
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, name="dense_layer_2"),
    ]
)
```



    
### Task 1.3 - Run
```
Model: "model_cnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (64, 26, 26, 32)          896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (64, 13, 13, 32)          0         
_________________________________________________________________
flatten (Flatten)            (64, 5408)                0         
_________________________________________________________________
dense_layer_2 (Dense)        (64, 2)                   10818     
=================================================================
Total params: 11,714
Trainable params: 11,714
Non-trainable params: 0
_________________________________________________________________
75/75 - 0s - loss: 1.9659e-05
Test loss: 1.965910632861778e-05

```
### Task 1.3 - Evaluate

Run the code multiple times and record the “average prediction error” for each run. Calculate the mean and standard deviation of the “average prediction error” over all runs

  - Run  1: 4.13e-03
  - Run  2: 2.76e-03
  - Run  3: 4.78e-03 
  - Run  4: 5.64e-03  
  - Run  5: 1.69e-03 
  - Run  6: 2.97e-03 
  - Run  7: 2.53e-03 
  - Run  8: 5.87e-03 
  - Run  9: 4.59e-03 
  - Run 10: 1.00e-03 

  - **Standard deviation**: 0.0017
  - **Mean**: 0.0036



Evaluate Make a comparison of the different models (i.e. Mθ, Mtrig, Mcnn) based on the the prediction accuracy on the test dataset and the number of trainable parameters. 


| Model   | Prediction accuracy | Trainable parameters  |
| ------------- |:-------------:      | -----:|
| Mθ            | 1.108             | 301,313   |
| Mtrig         | 0.023           |   301,442 |
| Mcnn          | 0.0036            |    11,714 |

The model.summary() function prints useful information about the model to the terminal. 
- Which model would you prefer and why? 

  Answer: The cnn model because it gives the best accuracy and requires the less trainable parameters. This model_cnn is the one that takes longer to run, but if computer resources are not a problem I still choosing it.
  
  If you change the activation of the last fully connected layer to ReLU, the prediction accuracy completely deteriorates. Why?
  
  Because RELU activation function doesn't let negative values, and therefore we are not able to predict negative sin and negative cos with this activation function
  
      <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/relu.png" width="500">
    

## Task 1.4
In the previous tasks, the prediction accuracy varied across different runs even though the underlying code and dataset remained unchanged. Uncomment the seed_experiment(...) line and change model_type to ‘model_theta’. Then, run the code to train model M θ several times while having the seed_value be either 0 or 1. 

### Task 1.4 - Evaluate


For both seed values, run the code multiple times and record the “average prediction error” for each run. Then, calculate the mean and standard deviation of the “average prediction error” over all runs for each seed value separately. What do you observe?


<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh"></th>
    <th class="tg-baqh" colspan="2">Run 1</th>
    <th class="tg-baqh" colspan="2">Run 2</th>
    <th class="tg-baqh" colspan="2">Run 3</th>
    <th class="tg-baqh" colspan="2">Run 4</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">seed_value</td>
    <td class="tg-baqh">mean</td>
    <td class="tg-baqh">std</td>
    <td class="tg-baqh">mean</td>
    <td class="tg-baqh">std</td>
    <td class="tg-baqh">mean</td>
    <td class="tg-baqh">std</td>
    <td class="tg-baqh">mean</td>
    <td class="tg-baqh">std</td>
  </tr>
  <tr>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0.3958</td>
    <td class="tg-baqh">0.0965</td>
    <td class="tg-baqh">0.3958</td>
    <td class="tg-baqh">0.0965</td>
    <td class="tg-baqh">0.3958</td>
    <td class="tg-baqh">0.0965</td>
    <td class="tg-baqh">0.3958</td>
    <td class="tg-baqh">0.0965</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1.5763</td>
    <td class="tg-baqh">0.9317</td>
    <td class="tg-baqh">1.5763</td>
    <td class="tg-baqh">0.9317</td>
    <td class="tg-baqh">1.5763</td>
    <td class="tg-baqh">0.9317</td>
    <td class="tg-baqh">1.5763</td>
    <td class="tg-baqh">0.9317</td>
  </tr>
</tbody>
</table>

If we set the seed value to 0 and repeat the experiment 10 times, we get a mean average prediction error of 0.39. If we set the seed value to 1 and repeat the experiment 10 times we get a mean average prediction error of 1.57 rad. This means that using a fixed seed value results in exactly the same model and errors when repeating the experiments. 


<figure>
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/seed_value_0_run1.png" width="400">
  <figcaption> Every time you run the model_theta with seed_value = 0, you get the same figure.</figcaption>
</figure>



---

<figure>
  <img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/seed_value_1_run1.png" width="400">
  <figcaption> Every time you run the model_theta with seed_value = 1, you get the same figure.</figcaption>
</figure>

---



What is the benefit of seeding the pseudo-random generator in practice?



If we do not fix the seed value, the weights of the networks are initialized differently every iteration and therefore result in different trained models.
The **weights of the network are initialized based on the pseudo random number that depends on the seed value**. The benefit of seeding the pseudo random number generator in practice, is that the **results will be reproducible**. This is relevant for example when sharing your code that is used for scientific experiments.
In this way, other researchers will obtain exactly the same results when running the code.








## Task 1.5
If our goal is to control the pendulum, the angular velocity θ̇ is generally also required. However, temporal information cannot be extracted from individual images. Describe a strategy (architecture, data pre-processing, etc...) to estimate both the angle and angular velocity from images.


We are given some sequential data as input, a sequence of raw image observations of the pendulum trajectories. A single image observation is not enough to provide any temporal information, for example, we cannot know in which direction is the pendulum going and its velocity. 
The strategy I propose is a combiantion of an autoencoder + RNN (recurrent neural network).
RNNs are able to embed past observations in their hidden state *h*. In particular I propose the use of LSTMs (Long short-term memory) which are an improved version of the simple vanilla RNN. 

Because the observations are high dimensional raw images, it is necessary to reduce its dimensionality in order to use them and extract the relevant information. For this, we can use an autoencoder (encoder + latent space + decoder) to compress the important features of the image into a latent space.

The output of the decoder is a prediction of the next observation, and a consequence of learning to predict this next observation, the model also learns to embed past observations in the hidden state *h* of the LSTM.


The following picture shows a general view of the architecture:

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/task1.5.png" width="500">

---

### Another answer for 1.5
One way to estimate the angular velocity, is by using observations that consist of a number of consecutive downsampled images. Since the observation now consists of images from consecutive time steps, the observation contains the velocity information that we would like to extract. Also, we probaly want to convert the RGB images to grey-scale images in order to reduce the dimensionality of the observation.
A possible architecture for predicting the angle and angle velocity is the following:

- The input consists of a stack of a number of consecutive downsampled grey-scale images.

- Next, a convolutional layer of rectified linear units will be used with an appropriate number of
filters and kernel sizes.

- After the convolutional layer, we can add a max pooling layer to reduce the dimensionality and
add basic translational invariance to the model.

- Now we will flatten the data, since the next steps do not depend on the spatial relationship of the
data.

- Finally, the flat layer will be connected to a fully-connected linear output layer.


It is also possible to use multiple convolutional layers of rectified linear units or to have a fully-connected layer of rectified linear units before the linear output layer. But this will increase the complexity of the model and therefore the number of trainable parameters.

### Yet another concept
If you process sequences of images and at time step _t_, the pendulum is at a certain position and at the next timestep _t+1_ it has move, it is possible from that difference to extract the velocity (angle sweep / time)


---
References:
- R. Perez-Dattari, C. Celemin, G. Franzese, J. Ruiz-del-Solar and J. Kober, "Interactive Learning of Temporal Features for Control: Shaping Policies and State Representations From Human Feedback," in IEEE Robotics & Automation Magazine, vol. 27, no. 2, pp. 46-54, June 2020, doi: 10.1109/MRA.2020.2983649.

- X. Zhao, X. Han, W. Su and Z. Yan, "Time series prediction method based on Convolutional Autoencoder and LSTM," 2019 Chinese Automation Congress (CAC), Hangzhou, China, 2019, pp. 5790-5793, doi: 10.1109/CAC48633.2019.8996842.


