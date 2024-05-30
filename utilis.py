import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def extract_var(data):
    # separate the variables from the input data
    x = data[:, 0:1]
    t = data[:, 1:2]
    return x, t


def jacobian(tape, pred, x, t):
    # first order derivative
    dx = tape.gradient(pred, x)
    dt = tape.gradient(pred, t)
    return dx, dt


def hessian(tape, dx, x):
    # second order derivative
    ddx = tape.gradient(dx, x)
    return ddx


def pde_loss(ux, ut, uxx, u, nu):
    # pde loss of the Burgers equation
    loss = ut + tf.math.multiply(u, ux) - nu * uxx
    return tf.reduce_mean(tf.math.square(loss))


def init_loss(model, input_init):
    # compute the loss of initial condition
    x_init, t_init = extract_var(input_init)
    u_init = model(tf.concat([x_init, t_init], axis=1))
    loss = u_init + tf.math.sin(np.pi * x_init)
    return tf.reduce_mean(tf.math.square(loss))


def bc_loss(model, input_bc1, input_bc2):
    # compute the loss of two boundaries
    xbc1, tbc1 = extract_var(input_bc1)
    xbc2, t_bc2 = extract_var(input_bc2)
    ubc1 = model(tf.concat([xbc1, tbc1], axis=1))
    ubc2 = model(tf.concat([xbc2, t_bc2], axis=1))
    loss = ubc1 + ubc2
    return tf.reduce_mean(tf.math.square(loss))


def loss_fn(model, ux, ut, uxx, u, nu, input_init, input_bc1, input_bc2):
    # compute the pde loss
    loss_pde = pde_loss(ux, ut, uxx, u, nu)
    # compute the initial condition loss
    loss_init = init_loss(model, input_init)
    # compute the boundary loss
    loss_bc = bc_loss(model, input_bc1, input_bc2)
    return loss_pde + loss_init + loss_bc


def prediction(model, data):
    # use the trained NN to predict u with test data set
    x, t = extract_var(data)
    for i in range(data.shape[0]):
        if i == 0:
            pred = model(tf.concat([x, tf.ones_like(t) * t[i, 0]], axis=1))
        else:
            p = model(tf.concat([x, tf.ones_like(t) * t[i, 0]], axis=1))
            pred = tf.concat([pred, p], axis=1)
        del i
    return pred


def plot(pred):
    ax = plt.gca()
    im = ax.imshow(pred, cmap='inferno', aspect=0.2, extent=[0, 1, -1, 1])
    plt.colorbar(im, shrink=0.6)
    plt.xlabel('Time: t')
    plt.ylabel('Domain: x')
    plt.show()
