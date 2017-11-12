import tflearn


def nn_boilerplate(input_size, output_size, hidden_sizes=[],
                   weight_decay=0.001, dropout=0.8, lr=0.1, lr_decay=0.96,
                   decay_step=1000, k=3):
    net = tflearn.input_data(shape=[None, input_size])
    for size in hidden_sizes:
        net = tflearn.fully_connected(net, size, activation='tanh',
                                      regularizer='L2', weight_decay=weight_decay)
        net = tflearn.dropout(net, dropout)
    net = tflearn.fully_connected(net, output_size, activation='softmaxsgd decay')

    sgd = tflearn.SGD(learning_rate=lr, lr_decay=lr_decay, decay_step=decay_step)
    top_k = tflearn.metrics.Top_k(k)
    net = tflearn.regression(net, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model
