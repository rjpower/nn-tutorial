A hastily assembled set of examples for building simple neural networks.

# How to use

If you're in a hurry: just read the example code to get a feel for how to use
tensorflow.  Otherwise, you can try out the extensions listed in the "Examples"
section below.

The tensorflow documentation is fairly extensive and will be useful as a
reference for these exercises: https://www.tensorflow.org/versions/r0.7/get_started/index.html

# Requirements

* tensorflow
* keras

These can be installed via pip (depending on your Python configuration, you may
need to prefix these commands with `sudo`).  On OS X, install the
homebrew version of Python first:

```
brew install python
```

```
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
pip install keras
```

Optional, but useful, is the Ipython notebook:

```
pip install notebook
```

# Examples

## scalar_regression.py

Builds possibly the simplest example we can make of gradient learning:
train a "network" to learn a linear predictor (a line): `y_p = w*x + b`.  Our
predictor tries to minimize the squared error between the actual `y` and our
predicted `y_p`.

We first build a dataset for testing our classifier.  The first array, `X` will
be our input feature to our classifier; we just initialize it with some
uniformly random values.

The second array, `Y` will be what our classifier is going to try to predict.
We initialize it as a linear function of X + some additional noise.

Now we use tensorflow to build up the abstract execution graph to perform `y_p =
weight*x + bias`.  Our loss function is just the squared error of our
prediction: we'll try to minimize this by adjusting our weights.

Finally, we create a new session to start training.  Iterate over the dataset 10
times (typically  a single pass through the training data is referred to as an
"epoch").  For each example we run the `train_op` to compute an estimate and
update our weights and bias.

Extensions:

* Hold out some training data to use as a verification set.  (The slice notation
  of Numpy should be useful here)
* Instead of minimizing the loss of a single example, try minimizing the mean
  loss.  What happens?
* Train a batch of 16 examples at a time instead of a single example.

## simple_classifier.py

Extends our simple regressor example in a few ways:

* Use softmax and cross-entropy to choose one of 2 labels
* Increase our input dimensionality
* Use multiple layers to improve classification performance
* Use non-linearities to allow learning of non-linear surfaces

As before, we define X and Y to be our training data, but now Y is a non-linear
function of X.  We also have converted from a regression problem to a
classification problem: we want to determine whether a given example lies inside
or outside of a sphere.

Extensions:

* Change the weight initialization from `truncated_normal` to uniform in (-1, 1)
* Try reducing the number of layers and see if you can get the same accuracy.
* Add an extra layer.
* Play with the different optimizers available.
* Remove the non-linearity between layers
