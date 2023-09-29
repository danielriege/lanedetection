import tensorflow as tf
import os

def train_model(model, data_gen_train, data_gen_valid, output_path, params, model_path):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    filepath_checkpoint = os.path.join(model_path, "checkpoint.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath_checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # get number of possible threads
    n_workers = os.cpu_count()

    history = model.fit(data_gen_train,
                validation_data=data_gen_valid,
                epochs=params.epochs,
                use_multiprocessing=False,
                workers=n_workers,
                callbacks=callbacks_list)
    return history