from data import *
from PINN import *

#############################################################
"Parameters to generate the data"
# maximum values for x, t
minval = [-1, 0]
# maximum values for x, t
maxval = [1, 1]
# size of the training set
ntrain = int(1e+3)
# number of points for the boundary, x = 1 and x = -1
nbc = 50
# number of points at t = 0
ninit = 100
# size of test set
ntest = 1000
# viscosity in the Burgers equatioin
nu = 0.01/np.pi
#############################################################
"Generate the training set and test set"
# generate the training set
input_train = train(ntrain, minval, maxval)
# generate the test set
input_test = test(ntest, minval, maxval)
"Generate the initial (t=0) and boundary (x=1 & x=-1) input data"
# data at t = 0 (initial condition)
input_init = data_init(ninit, -1, 1)
# data at x = 1 and x = -1
input_bc1 = data_bc1(nbc, 0, 1)
input_bc2 = data_bc2(nbc, 0, 1)
#############################################################
"Hyperparameters"
# number of units in each layer including the input/output layers
dim = [input_train.shape[1], 16, 32, 64, 1]
# optimizer used in PINN
opt = tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-8)
# number of epochs used in the training process
epoch = 10000
#############################################################
model = tf.keras.models.load_model('my_model.keras')
model.summary()
