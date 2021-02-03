> 💡

## Installation procedure

In REAME.md, line 11, add what you mean by move into that directory. Example cd ..
Line 27 Readme should be: irene@irene-computer:~/Dropbox/KBCS-Practical Assignment/KBCS-assignment-main$  Important: The folder is called KBCS-assignment

Tutorial for anaconda in Ubuntu 20.04: https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/
Important once you finish fllowing the instructions on how to install anaconda, close and open a new terminal!

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
## Task 1.2
## Task 1.3
## Task 1.4



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
