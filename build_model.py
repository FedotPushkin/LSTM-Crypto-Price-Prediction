from tensorflow import keras
from keras_tuner import HyperParameters, BayesianOptimization
import keras.regularizers as reg
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention  # , AlphaDropout
from keras.optimizers import Adam, Adamax, Adagrad


def build_model(x_adj, regression, learning_rate):

    reg = 'l2'
    # first layer
    model = Sequential()
    model.add(LSTM(50,
                   input_shape=(x_adj.shape[1], x_adj.shape[2]),
                   return_sequences=True,

                   recurrent_regularizer=reg,
                   kernel_regularizer=reg))   # recurrent_dropout=0.05))
    model.add(BatchNormalization())
    #model.add(Dropout(0.05))

    # second layer
    model.add(LSTM(100, return_sequences=True, recurrent_regularizer=reg, kernel_regularizer=reg))# kernel_regularizer=reg.L1(0.001)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.05))
    model.add(LSTM(100, return_sequences=False, recurrent_regularizer=reg, kernel_regularizer=reg))  # , recurrent_regularizer='l2'))# , kernel_regularizer=reg.L1(0.005)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Dense(50, activation='relu', kernel_regularizer=reg))
    # model.add(AlphaDropout(0.5))
    model.add(Dropout(0.1))
    # fourth layer and output
    # model.add(Dense(1, activation='linear'))    # kernel_regularizer=regularizers.L1(0.01)

    #optimizer = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.004)


    if regression:
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      metrics=[keras.metrics.RootMeanSquaredError()],
                      optimizer='nadam')
    else:
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'],
                      optimizer='rmsprop')

    return model


def build_model_bayesian(hp):
    model = Sequential()
    activation = 'tanh'
    reg = 'l2'
    model.add(keras.layers.LSTM(units=hp.Int('units1', min_value=75, max_value=75, step=5),
                                dropout=0,
                                #dropout=hp.Float('droput1', min_value=0.00, max_value=0.1, step=0.01),
                                #recurrent_dropout=hp.Float('redropout', min_value=0.94, max_value=0.94, step=0.01),
                                activation=activation,
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                return_sequences=True,
                                input_shape=(10, 13)))
    #Attention()
    model.add(BatchNormalization())
    model.add(keras.layers.LSTM(units=hp.Int('units2', min_value=165, max_value=165, step=5),
                                dropout=0,
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                #dropout=hp.Float('dropout2', min_value=0.00, max_value=0.1, step=0.01),
                                activation=activation, return_sequences=True))
    #Attention()
    model.add(BatchNormalization())
    model.add(keras.layers.LSTM(units=hp.Int('units3', min_value=165, max_value=165, step=5),
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                activation=activation))
    #Attention()
    model.add(BatchNormalization())
    model.add(Dense(units=hp.Int('units4_dense', min_value=100, max_value=100, step=10),
                    activation='relu',
                    kernel_regularizer=L2(hp.Choice('L2_1', [1e-4]))))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))

    optimizer_choice = hp.Choice('optimizer', ['adam'])
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 2*1e-3, 2*1e-3, sampling='log'))
    elif optimizer_choice == 'adamax':
        optimizer = Adamax(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    else:
        optimizer = Adagrad(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def opt_search(x_adj, y_adj, xval, yval):
    checkpoint_filepath = f'checkpoint_opt.weights.h5'
    monitor = 'val_loss'
    mode = 'max'
    model_checkpoint_callback = ModelCheckpoint(

        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    hp = HyperParameters()
    bayesian_opt_tuner = BayesianOptimization(
        build_model_bayesian,
        objective='val_loss',
        max_trials=1,
        num_initial_points=3,
        executions_per_trial=1,
        project_name='timeseries_bayes_opt_POC',
        overwrite=True, )

    bayesian_opt_tuner.search(x=x_adj, y=y_adj,
                              epochs=120,
                              validation_data=(xval, yval),
                              validation_split=0.8,
                              validation_steps=30,
                              steps_per_epoch=30,
                              callbacks=[model_checkpoint_callback,
                                         keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       patience=5,
                                                                       verbose=1,
                                                                       restore_best_weights=True),
                                         keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                           factor=0.5,
                                                                           patience=3,
                                                                           verbose=1,
                                                                           min_delta=1e-5,
                                                                           mode='min')])
    best_hps = bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = bayesian_opt_tuner.get_best_models()[0]
    best_model.save('best_opt_model.keras')
    bayesian_opt_tuner.results_summary(num_trials=10)
    print('Best lr:', best_hps.get('learning_rate'))
    print('Best optimizer:', best_hps.get('optimizer'))
    print('Best L2:', best_hps.get('L2_1'))
