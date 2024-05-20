from tensorflow.python.client import device_lib
from build_model import build_model
import platform
import tensorflow as tf
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
def plothistories(histories, y_pred_p, yval_p):
    for history in histories:
        # summarize history for accuracy

        # summarize history for loss

        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')


        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # fpr_keras, tpr_keras, thresholds_keras = roc_curve(yval_p.T[0], y_pred_p.T[0])

        # auc_keras = auc(fpr_keras, tpr_keras)

        #plt.figure(3)
        #plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        # plt.plot(fpr_keras, tpr_keras, label='RF (area = {:.3f})'.format(auc_keras))
        #plt.xlabel('False positive rate')
        #plt.ylabel('True positive rate')
        #plt.title('ROC curve')
        #plt.legend(loc='best')
        # plt.show()

def shuffle_and_train(x_adj, y_adj, tag, reshuffle):
    # count the number of each label
    # count_1 = np.count_nonzero(y_adj)
    # count_0 = y_adj.shape[0] - count_1
    # cut = min(count_0, count_1)
    use_checkpoints = True
    # save some data for testing
    # train_idx = int(cut * split)
    if reshuffle:
        # balance 50/50 and shuffle pos and neg examples
        np.random.seed(42)
        shuffle_index = np.random.permutation(x_adj.shape[0])
        x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]

        # find indexes of each label
        idx_1 = np.argwhere(y_adj == 1).flatten()
        idx_0 = np.argwhere(y_adj == 0).flatten()

        shuffle_1 = np.random.permutation(len(idx_1))
        shuffle_0 = np.random.permutation(len(idx_0))
        minlen = min(len(idx_1), len(idx_0))
        if len(idx_1) > len(idx_0):
            idx_1 = idx_1[shuffle_0]
        else:
            idx_0 = idx_0[shuffle_1]
        # shuffle_index = np.random.permutation(minlen)

        # grab specified cut of each label put them together
        x_adj = np.concatenate((x_adj[idx_1[:minlen]], x_adj[idx_0[:minlen]]), axis=0)
        # X_test = np.concatenate((x_adj[idx_1[train_idx:cut]], x_adj[idx_0[train_idx:cut]]), axis=0)
        y_adj = np.concatenate((y_adj[idx_1[:minlen]], y_adj[idx_0[:minlen]]), axis=0)
        # y_test = np.concatenate((y_adj[idx_1[train_idx:cut]], y_adj[idx_0[train_idx:cut]]), axis=0)

        # shuffle again to mix labels
        np.random.seed(42)
        cp = Counter(y_adj)
        print(f'balanced pos {cp.most_common(2)}')
        shuffle_index = np.random.permutation(x_adj.shape[0])
        x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]
        bal = Counter(y_adj)
        print(f'balanced examples: {bal.most_common(2)}')
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print(f'available devices: {get_available_devices()}')
    if gpus:
        if platform.system() == "Windows":
            print(f'will compute on {tf.test.gpu_device_name()}')
            #tf.config.set_logical_device_configuration(
            #    gpus[0],
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=2800)]
            #)


    #tf.debugging.set_log_device_placement(True)
    #strategy = tf.distribute.MirroredStrategy(cpus)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    histories = list()
    #with strategy.scope():

    # for index, (train_indices, val_indices) in enumerate(skf.split(x_adj, y_adj)):
    # following code is for kfold
    for index in range(1):
        print("Training on fold " + str(index + 1) + "/5...")
        # Generate batches from indices
        if True:    # index == 0:
            if reshuffle:
                # bins = np.linspace(min(y_adj), max(y_adj), 5)
                # y_binned = np.digitize(y_adj, bins)

                xtrain, xval, ytrain, yval = train_test_split(x_adj, y_adj, test_size=0.2, shuffle=True)
                # stratify=y_binned)
                # xtrain, xval = [train_indices], x_adj[val_indices]
                # ytrain, yval = y_adj[train_indices], y_adj[val_indices]
                # ytrain, yval = to_categorical(ytrain, 2), to_categorical(yval, 2)
                # graph(xtrain[0], hold_g=ytrain.T[0], start_g=timesteps, len_g=50, col_type=0)
                np.save(f'xtrain_{tag}', xtrain, allow_pickle=True)
                np.save(f'ytrain_{tag}', ytrain, allow_pickle=True)
                np.save(f'xval_{tag}', xval, allow_pickle=True)
                np.save(f'yval_{tag}', yval, allow_pickle=True)

            xtrain, xval = np.load(f'xtrain_{tag}.npy', allow_pickle=True), \
                np.load(f'xval_{tag}.npy', allow_pickle=True)
            ytrain, yval = np.load(f'ytrain_{tag}.npy', allow_pickle=True), \
                np.load(f'yval_{tag}.npy', allow_pickle=True)

            ytrain, yval = to_categorical(ytrain, 2), to_categorical(yval, 2)
            for lr in range(1, 2):
                model = build_model(xtrain, learning_rate=0)    # 0.009/(pow(2, lr)))
                checkpoint_filepath = f'checkpoint_{tag}_{index}.weights.h5'
                if os.path.isfile(checkpoint_filepath) and use_checkpoints:
                    model.load_weights(checkpoint_filepath)
                model_checkpoint_callback = ModelCheckpoint(

                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)
                with tf.device('/gpu:0'):
                    history = model.fit(xtrain, ytrain,
                                        epochs=10,
                                        batch_size=1024,
                                        shuffle=True,
                                        validation_data=(xval, yval),
                                        callbacks=[model_checkpoint_callback])
                model.load_weights(checkpoint_filepath)
                y_pred_p = model.predict(xval)
                histories.append(history)
                plothistories([history], y_pred_p, yval)
            # model.save(f'models/lstm_model_{tag}_{index}.h5', save_format='h5')
            model.save('my_model.keras')

    ind = 0
    for history in histories:
        np.save(f'histories_{tag}_{ind}', history.history['loss'], allow_pickle=True)
        np.save(f'histories_{tag}_{ind}', history.history['val_loss'], allow_pickle=True)
        ind += 1
    #   X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    #   X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

