

## Installation procedure
> 💡 The folder I download with the files it is called *KBCS-assignment-main* and not *KBCS-assignment* as indicated in the `README.md` file.

I followed this tutorial for installing anaconda on Ubuntu **20.04**: https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/

⚠️ Important once you finish following the instructions on the tutorial, close and open a new terminal as it is indicated at the end of the tutorial.



<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_1.png" width="700">

Once it is installed, you will see a message on your terminal telling you how to activate the virtual environment:

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_2.png" width="700">
Go and activate the virtual environment: 


```bash

conda activate kbcs

```
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/Installation_procedure_3.png" width="700">

Once inside the virtual environment, (*Note: you are inside the virtual environment if you see (kbcs) in front of the terminal line*), run if you wish the file `generate_data.py`, it takes some time:

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/generate_data_1.png" width="700">

<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/generate_data_2.png" width="700">



<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/radians.png" width="200">
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/circle_20.jpg" width="200">

In REAME.md, line 11, add what you mean by move into that directory. Example cd ..
Line 27 Readme should be: irene@irene-computer:~/Dropbox/KBCS-Practical Assignment/KBCS-assignment-main$  Important: The folder is called KBCS-assignment


Maybe include a picture of the angles of the pendulum or a more detailed description like, from upright position, going right that means positive pi...

IN the Readme, in the training data sectionn "which we concatenate into a single sequence and save as a single dataset", here the word sequence, maybe put trajecory?

In the train file maybe already include the header keras and layers?
Maybe is confusing, if you add the headers then, tf.keras.layers?

Maybe this link is useful for the compile step? https://keras.io/api/optimizers/

It was intentional to already include how to compile the model?

"Make sure that we have a train-validation split of 0.2 and a batch size of 64", add something like, this is already added in the provided train.py file

In the run section, what is the meaning of "Close the generated plots to start a live simulation of your prediction model in case you run locally."

Say that change model_type is around lie 63

CNN
"kernel_size 	An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions. "


## Task 1.1


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
> 💡 It was intentional to already include how to compile the model?


### Task 1.1 - Train
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
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/average_error_model_theta.png" width="700">
<img src="https://github.com/irenebosque/KBCS-Practical-Assignment/blob/main/images/pendulum_model_theta.png" width="700">

Average predicion error $2.33$
<img src="https://render.githubusercontent.com/render/math?math=2.33*10^0">
<img src="https://render.githubusercontent.com/render/math?math=\pi">
## Task 1.2



## Task 1.3
## Task 1.4

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
