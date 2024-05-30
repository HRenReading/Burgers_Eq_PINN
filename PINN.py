from tensorflow.keras.layers import Dense, Input
import time
from utilis import *


class Network:
    @classmethod
    def network(cls, input_dim):
        """
        The neural network architecture in physics informed neural network.

        :param input_dim: number of unites in each layer
        :return: the neural network
        """
        # input layer
        x_input = Input(shape=(input_dim[0],))
        # hidden layers
        for i in range(1, len(input_dim) - 1):
            if i == 1:
                x = Dense(units=input_dim[i], activation='tanh',
                          kernel_initializer='glorot_normal',
                          bias_initializer='zeros')(x_input)
            else:
                x = Dense(units=input_dim[i], activation='tanh',
                          kernel_initializer='glorot_normal',
                          bias_initializer='zeros')(x)
            del i
        # output layer
        x = Dense(units=input_dim[-1], activation=None,
                  kernel_initializer='glorot_normal',
                  bias_initializer='zeros')(x)

        model = tf.keras.Model(inputs=x_input, outputs=x)
        return model


class Gradient:
    @classmethod
    def grad(cls, model, data):
        x, t = extract_var(data)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(t)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                pred = model(tf.concat([x, t], axis=1))
                ux, ut = jacobian(g, pred, x, t)
            uxx = hessian(gg, ux, x)
        return ux, ut, uxx, pred


def pinn_train(dim, input_train, nu, input_init, input_bc1, input_bc2, epoch, opt):
    """
    Train the neural network with PINN algorithm using training set.

    :param dim: number of unites in each layer
    :param input_train: training set
    :param nu: viscosity parameter
    :param input_init: initial condition data
    :param input_bc1: upper boundary condition
    :param input_bc2: lower boundary condition
    :return: the trained model
    """
    start_time = time.time()
    # initialize the model
    model = Network.network(dim)
    # train the model
    for i in range(epoch + 1):
        with tf.GradientTape(persistent=True) as tape:
            ux, ut, uxx, u = Gradient.grad(model, input_train)
            loss = loss_fn(model, ux, ut, uxx, u, nu, input_init, input_bc1, input_bc2)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if i % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(i, loss.numpy()))
        del i
    print("Training time: %s seconds" % (time.time() - start_time))
    return model
