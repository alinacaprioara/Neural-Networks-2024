# Flappy Bird
## Training a Neural Network using Q-learning Algorithm to control an agent in the Flappy Bird game
### Team Members:
- **Căprioară Alina** 
- **Neagu Alex-Ștefan**

## Introduction

This report provides the architecture, hyperparameters and experimentation attempts to optimize the process of maximizing final reward.
As setup, we used the flappy-bird-gymnasium environment.

### [[Demo Video]](https://youtu.be/jPBf9_DFl5Y)
## Architecture

### I. First Attempt - Outline image preprocessing

### Workflow
1. Environment initialization
2. Frame preprocessing and stacking
3. Action selection
4. Frame jumping if action="flap"
5. Observing for OBSERVE number of episodes and then training the neural network.

###  Neural Network

We used a Convolutional Neural Network (CNN) with the purpose of processing the game stacked frames and predicting the Q-values for each action.

1. **Input Layer**: 4 stacked frames of the game (84x84x4)
2. **Convolutional Layers**:
   - **Layer 1**: Conv2D with 32 filters, kernel size 7x7, stride 3, and ReLU activation.
   - **Layer 2**: Conv2D with 64 filters, kernel size 5x5, stride 2, and ReLU activation.
   - **Layer 3**: Conv2D with 64 filters, kernel size 3x3, stride 1, and ReLU activation.
3. **Fully Connected Layers**:
    - **Fully Connected Layer 1**: 256 neurons and ReLU activation.
    - **Fully Connected Layer 2**: 256 neurons with output of action space size (2 actions: flap/do nothing)

The output consists of the Q-values for each action(flap/do nothing).


### Image Preprocessing

The RGB frame obtained by rendering the game is preprocessed for optimized training.
- **Grayscale Conversion**
- **Edge Detection Using Conv2d**: To detect the outline of the objects in the frame in order to simplify the image and accentuate the important features, each frame passes through an edge detection filter, using the outline_kernel.
```
outline_kernel = [[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]]
```

- **Resizing**: The frames are resized to 84x84.
- **Normalization**: The pixel values are normalized to the range [0, 1].

<img src="Report Images/image_preprocess_1.png" alt="Image preprocessing" width="250">


### Hyperparameters

| Hyperparameter     | Value  |
|--------------------|--------|
| Learning Rate      | 0.01   |
| Gamma              | 0.99   |
| Batch Size         | 32     |
| Initial Epsilon    | 1.0    |
| Final Epsilon      | 0.01   |
| Epsilon Decay      | 0.995  |
| OBSERVE            | 1000   |
| EXPLORE            | 10000  |
| Replay Buffer Size | 5000   |

### Results

<img src="Report Images/first_attempt_1.png" alt="Results 1" width="400">
<img src="Report Images/first_attempt_2.png" alt="Results 2" width="400">
<img src="Report Images/first_attempt_3.png" alt="Results 3" width="400">


#

### II. Second attempt - Dual Q-Learning

We tried implementing Dual Q-Learning, with 2 networks: QA and QB.
The idea mainly consists in using QB_target to calculate the target Q-value when QA selects the action, and vice versa. 

This alternation helps stabilize the training, as the action selection and target evaluation are being done by different networks.

We noticed slower training in this experiment, thus returning to the original Q-learning algorithm.


<img src="Report Images/dual_attempt.png" alt="Dual attempt results" width="400">

#

###  III. Final Attempt - Best version - Improved image processing

We improved the image processing by implementing a black and white conversion.
The outline of the objects was set to white, while everything else was set to black.
Another improvement in this attempt was cropping the ground.
These changes added a significant improvement in the training process, as the frames were now simplified, better suited for training.

<img src="Report Images/image_process_2.png" alt="Image preprocessing" width="250">

Training routine for the current model:
1. 1000 epochs for exploring (very quick)
2. ~300 epochs for learning
3. Restart learning, keep model from epoch 1300
4. 1000 epochs for exploring (very quick)
5.  ~300 epochs for learning
6. Restart learning, keep model from epoch 1300
7. 1000 epochs for exploring (very quick)
8. ~600 epochs for learning


### Hyperparameters

| Hyperparameter     | Value |
|--------------------|-------|
| Learning Rate      | 0.001 |
| Gamma              | 0.99  |
| Batch Size         | 32    |
| Initial Epsilon    | 0.1   |
| Final Epsilon      | 0.001 |
| OBSERVE            | 1000  |
| EXPLORE            | 10000 |
| Replay Buffer Size | 50000 |

### Results
With this model we achieved the best score of 249 passed pipes.

<img src="Report Images/best_score.png" alt="Best score" width="400">

### IV. Other Attempts - Failed and Succesful? Experiments

The previous models were what we found to have a moderate level of succes, and they were our main focus.

One area we tried to tackle was the amount of points the agent got in a run. We noticed that at times, the Agent could get less points when passing the first pipe, and more if the bird had more up-time.

A solution to this issue would be increasing the number of points the agent got by passing a pipe, making it more bias towards passing pipes.

We have no data wether or not this solution is efficient, but we noticed a small increase in convergence.

#### Sarsa Agent

After using an off-policy agent for most of our testing, we noticed that the main cause of failure was hitting the bottom pipe. This was happening because or model prioritized the shortest path, thus jumping at the exact moment when it was needed.

SARSA, an on-policy algorithm, displayed a different behaviour, the bird trying to always be at the center of the two pipes.

### Conclusions

We noticed the first 1000 iterations in which the agent only explores the environment and doesn't train are important for the training.

For our best model(*the last one*) we noticed that even though it achieves great results on average, it still lacks in consistency. We think that this can be further improved by increasing the training times of the model
