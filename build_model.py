from tensorflow import keras
from keras_tuner import HyperParameters, BayesianOptimization
import keras.regularizers as reg
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention  # , AlphaDropout
from keras.optimizers import Adam, Adamax, Adagrad, RMSprop


def build_model(x_adj, regression, learning_rate, features):

    reg = 'l2'
    # first layer
    model = Sequential()
    model.add(LSTM(120,
                   input_shape=(x_adj.shape[1], features),
                   return_sequences=True,
                   dropout=0,
                   #recurrent_dropout=0.8,
                   recurrent_regularizer=reg,
                   kernel_regularizer=reg))   # recurrent_dropout=0.05))
    model.add(BatchNormalization())
    #model.add(Dropout(0.05))

    # second layer
    model.add(LSTM(150, return_sequences=True,
                   dropout=0.1,
                   #recurrent_dropout=0.8,
                   recurrent_regularizer=reg,
                  kernel_regularizer=reg))# kernel_regularizer=reg.L1(0.001)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.05))
    model.add(LSTM(190, return_sequences=False,
                   dropout=0.1,
                   recurrent_regularizer=reg,
                   kernel_regularizer=reg))  # , recurrent_regularizer='l2'))# , kernel_regularizer=reg.L1(0.005)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Dense(200, activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization())
    # model.add(AlphaDropout(0.5))
    model.add(Dropout(0.1))
    # fourth layer and output
    # model.add(Dense(1, activation='linear'))    # kernel_regularizer=regularizers.L1(0.01)

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)


    if regression:
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError()],
                      optimizer='nadam')
    else:
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'],
                      optimizer=optimizer)

    return model


def build_model_bayesian(hp):
    model = Sequential()
    activation = 'tanh'
    reg = 'l2'
    model.add(keras.layers.LSTM(units=hp.Int('units1', min_value=80, max_value=120, step=10),
                                #dropout=0,
                                dropout=hp.Float('droput1', min_value=0.00, max_value=0.2, step=0.05),
                                #recurrent_dropout=hp.Float('redropout', min_value=0.8, max_value=0.8, step=0.1),
                                activation=activation,
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                return_sequences=True,
                                input_shape=(10, 13)))
    #Attention()
    model.add(BatchNormalization())
    model.add(keras.layers.LSTM(units=hp.Int('units2', min_value=150, max_value=190, step=10),
                                #dropout=0,
                                recurrent_regularizer=reg,
                                #recurrent_dropout=hp.Float('redropout', min_value=0.00, max_value=0.9, step=0.1),
                                kernel_regularizer=reg,
                                dropout=hp.Float('dropout2', min_value=0.00, max_value=0.2, step=0.05),
                                #return_sequences=True,
                                activation=activation,
                                ))
    #Attention()
    #model.add(BatchNormalization())
    #model.add(keras.layers.LSTM(units=hp.Int('units3', min_value=40, max_value=160, step=10),
    #                            recurrent_regularizer=reg,
    #                            kernel_regularizer=reg,
    #                            activation=activation))
    #Attention()
    model.add(BatchNormalization())
    model.add(Dense(units=hp.Int('units4_dense', min_value=120, max_value=200, step=10),
                    activation='relu',
                    kernel_regularizer=L2(hp.Choice('L2_1', [1e-4, 1e-3, 1e-2]))))

    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', min_value=0.00, max_value=0.25, step=0.05)))
    model.add(Dense(2, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', ['rms'])
    if optimizer_choice == 'rms':
        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', 2*1e-3, 2*1e-3, sampling='log'))
    elif optimizer_choice == 'adamax':
        optimizer = Adamax(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    else:
        optimizer = Adagrad(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def opt_search(x_adj, y_adj, xval, yval):
    checkpoint_filepath = f'timeseries_bayes_opt_POC/trial_00/checkpoint'
    monitor = 'val_loss'
    mode = 'min'
    model_checkpoint_callback = ModelCheckpoint(

        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    hp = HyperParameters()
    trials = 20
    bayesian_opt_tuner = BayesianOptimization(
        build_model_bayesian,
        objective='val_loss',
        max_trials=trials+5,
        num_initial_points=3,
        executions_per_trial=1,
        project_name='timeseries_bayes_opt_POC',
        overwrite=True, )

    bayesian_opt_tuner.search(x=x_adj, y=y_adj,
                              epochs=150,
                              validation_data=(xval, yval),
                              validation_split=0.8,
                              validation_steps=30,
                              steps_per_epoch=30,
                              callbacks=[model_checkpoint_callback,
                                         keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       patience=12,
                                                                       verbose=1,
                                                                       #start_from_epoch=3,
                                                                       restore_best_weights=True
                                                                       ),
                                         keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                           factor=0.25,
                                                                           patience=5,
                                                                           verbose=1,
                                                                           min_delta=1e-5,
                                                                           mode='min')])

    best_model = bayesian_opt_tuner.get_best_models(trials)
    for i in range(trials):
        best_model[i].save(f'best_opt_model_{i}.keras')
    bayesian_opt_tuner.results_summary(num_trials=trials)
    best_hps = bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Best lr:', best_hps.get('learning_rate'))
    print('Best optimizer:', best_hps.get('optimizer'))
    print('Best L2:', best_hps.get('L2_1'))
