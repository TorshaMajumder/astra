import tensorflow as tf
import logging
import os, sys
from core.data  import pretraining_records, create_dataset

from core.output    import ProjectionHead
from core.tboard    import save_scalar, draw_graph
from core.losses    import custom_rmse, custom_bce, nt_bxent_loss
from core.metrics   import custom_acc
from core.encoder   import Encoder
from core.scheduler import CustomSchedule, warmup_schedule

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tqdm import tqdm
import random
import numpy as np
import itertools
import pickle 
from astropy.cosmology import WMAP9 as cosmo

tf.config.run_functions_eagerly(True)

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')


def calc_luminosity(flux, mu):
    """ Normalise flux light curves with distance modulus.

    Parameters
    ----------
    flux : array
        List of floating point flux values.
    fluxerr : array
        List of floating point flux errors.
    mu : float
        Distance modulus from luminosity distance.

    Returns
    -------
    fluxout : array
        Same shape as input flux.
    fluxerrout : array
        Same shape as input fluxerr.

    """

    d = 10 ** (mu/5 + 1)
    dsquared = d**2
    norm = 1e18
    # flux = (flux_old - np.nanmean(flux_old))/(np.nanmax(flux_old) - np.nanmin(flux_old))
    # fluxerr = fluxerr_old/(np.nanmax(fluxerr_old) - np.nanmin(fluxerr_old))

    fluxout = flux * (4 * np.pi * dsquared/norm)
    # fluxerrout = fluxerr * (4 * np.pi * dsquared/norm)

    # fluxout = (fluxout - np.nanmean(fluxout))/(np.nanmax(fluxout) - np.nanmin(fluxout))
    # fluxerrout = fluxerrout/(np.nanmax(fluxerrout) - np.nanmin(fluxerrout))



    return fluxout

def correct_time_dilation(t, redshift):

  t = t / (1 + redshift)

  return t


def correct_for_distance(flux, redshift):

  dlmu = cosmo.distmod(redshift).value
  flux_ = calc_luminosity(flux, dlmu)

  return flux_


def add_redshift(flux, time):

    redshift_ = np.random.uniform(0.001, 0.5)
    time_dilation = correct_time_dilation(time, redshift_)

    # data['time'] = time_dilation
    flux_ = correct_for_distance(flux, redshift_)

    return time_dilation, flux_

            


def add_white_noise(ip, redshift=False):

    len_ = ip["length"].numpy()[0]
    masked_features= ip["input"].numpy()
    flux = masked_features[:, :(len_), 1]
    masked_time = ip["times"].numpy()
    time = masked_time[:, :(len_), :]
    
    if redshift:
        time, flux = add_redshift(flux, time)
        masked_time[:, :(len_), :] = time
        ip["times"] = tf.convert_to_tensor(masked_time)
    
    rand_int = random.randint(1, 50)
    noise = np.random.normal(0.01, rand_int, len_)
    noise = noise.reshape((1, len_))
    
    flux_noise = flux + noise
    masked_features[:, :(len_), 1] = flux_noise
    ip["input"] = tf.convert_to_tensor(masked_features)
    return ip

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 meta_shape=54,
                 emd_dim=256,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 no_train=True,
                 maxlen=100,
                 batch_size=None):

    serie  = Input(shape=(maxlen, 3),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')
    length = Input(shape=(maxlen,),
                  batch_size=None,
                  dtype=tf.int32,
                  name='length')
    meta = Input(shape=(54,),
                  batch_size=None,
                  name='meta')            

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times,
                    'meta':meta,
                   'length':length}

    encoder = Encoder(num_layers,
                d_model,
                meta_shape,
                emd_dim,
                maxlen,
                num_heads,
                dff,
                base=base,
                rate=dropout,
                use_leak=use_leak,
                name='encoder')

    if no_train:
        encoder.trainable = False

    x = encoder(placeholder)

    x = ProjectionHead(name='ProjectionHead')(x)

    return Model(inputs=placeholder,
                 outputs=x,
                 name="ASTROMER")

@tf.function
def train_step(model, x1, x2, x3, batch_size, temperature, opt):
    N = 3*batch_size
    positive_pairs_list = list()
    with tf.GradientTape() as tape:
        ### -----**CHANGED**----- ###
        # x_pred = model(batch)
        # mse = custom_rmse(y_true=batch['output'],
        #                  y_pred=x_pred,
        #                  mask=batch['mask_out'])
        ### -----**CHANGED**----- ###
        # N = 3*batch_size
        # positive_pairs_list = list()
        #
        z1 = model(x1)
        z2 = model(x2)
        z3 = model(x3)
        # print(z1.shape)
        z= tf.concat([z1, z2, z3], axis=0)
        
        # z = tf.squeeze(z, axis=-1)
        
        iter_ = z1.shape[0]
        # z= tf.concat([tf.squeeze(z1, axis=-1), tf.squeeze(z2, axis=-1), tf.squeeze(z3, axis=-1)], axis=0)
        # print(z1.shape)
        # z = tf.Variable(tf.zeros((z1.shape[0] + z2.shape[0] + z3.shape[0], z1.shape[1]), dtype= tf.float32))
        # z[0::3].assign(tf.squeeze(z1, axis=-1))
        # z[1::3].assign(tf.squeeze(z1, axis=-1))
        # z[2::3].assign(tf.squeeze(z1, axis=-1))
        # print(z.shape)
        for i in range(iter_):
            starting_index = i + (iter_*np.arange(3))
            positive_pairs_list.extend(list(itertools.product(starting_index, repeat=2)))
        positive_pairs = tf.constant(list(positive_pairs_list))

        loss = nt_bxent_loss(z, positive_pairs, temperature)
        loss = loss/N
        # print(loss)

    # print(model.trainable_variables)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

@tf.function
def valid_step(model, x1, x2, x3, batch_size, temperature, opt):
    # model, batch, return_pred=False, normed=False):
    N = 3*batch_size
    positive_pairs_list = list()
    with tf.GradientTape() as tape:
        ### -----**CHANGED**----- ###
        #     x_pred = model(batch)
        #     x_true = batch['output']
        #     mse = custom_rmse(y_true=x_true,
        #                       y_pred=x_pred,
        #                       mask=batch['mask_out'])

        # if return_pred:
        #     return mse, x_pred, x_true
        ### -----**CHANGED**----- ###
        
        #
        z1 = model(x1)
        z2 = model(x2)
        z3 = model(x3)
        z= tf.concat([z1, z2, z3], axis=0)
        # z = tf.squeeze(z, axis=-1)
        iter_ = z1.shape[0]
        # print(z1.shape)
        # z = tf.Variable(tf.zeros((z1.shape[0] + z2.shape[0] + z3.shape[0], z1.shape[1]), dtype= tf.float32))
        # z[0::3].assign(tf.squeeze(z1, axis=-1))
        # z[1::3].assign(tf.squeeze(z2, axis=-1))
        # z[2::3].assign(tf.squeeze(z3, axis=-1))
        for i in range(iter_):
            starting_index = i + (iter_*np.arange(3))
            positive_pairs_list.extend(list(itertools.product(starting_index, repeat=2)))
        positive_pairs = tf.constant(list(positive_pairs_list))

        loss = nt_bxent_loss(z, positive_pairs, temperature)
        loss = loss/N

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, path='main-code/data/new_data/ZTF/records/ZTF/r-LCs/',
          batch_size=50,
          temperature=1.0,
          patience=20,
          exp_path='main-code/presentation/experiments/test',
          epochs=1,
          finetuning=False,
          use_random=True,
          num_cls=2,
          lr=1e-3,
          redshift=False,
          maxlen=200,
          verbose=1):

    os.makedirs(exp_path, exist_ok=True)

    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'valid'))
    ### -----**CHANGED**----- ###
    # batch = [t for t in train_dataset.take(1)][0]
    # draw_graph(model, batch, train_writter, exp_path)
    ### -----**CHANGED**----- ###
    
    # Optimizer
    # lr_scheduler = tf.keras.optimizers.schedules.LearningRateSchedule(warmup_schedule, verbose=1)
    # optimizer = tf.keras.optimizers.YourOptimizer(learning_rate=lr_scheduler)

    custom_lr = CustomSchedule(model.get_layer('encoder').d_model)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=custom_lr,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    ### -----**CHANGED**----- ###
    # To save metrics
    # train_mse  = tf.keras.metrics.Mean(name='train_mse')
    # valid_mse  = tf.keras.metrics.Mean(name='valid_mse')
    ### -----**CHANGED**----- ###

    # Training Loop
    best_loss = 999999.
    es_count = 0
    step_wise_train_loss = []
    step_wise_val_loss = []
    epoch_wise_train_loss = []
    epoch_wise_val_loss = []
    pbar = tqdm(range(epochs), desc='epoch')
    for epoch in pbar:
        

        x1 = pretraining_records(os.path.join(path, 'train'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.0,
                                        rnd_frac=1.0,
                                        same_frac=0.1)

        x2 = pretraining_records(os.path.join(path, 'train'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.2,
                                        rnd_frac=1.0,
                                        same_frac=0.1)

        x3 = pretraining_records(os.path.join(path, 'train'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.35,
                                        rnd_frac=1.0,
                                        same_frac=0.15)
        
        
        
        for i, j, k in zip(x1, x2, x3):

            i_ = add_white_noise(i, redshift=redshift)
            j_ = add_white_noise(j, redshift=redshift)
            k_ = add_white_noise(k, redshift=redshift)

            train_loss = train_step(model, i, j, k, 
                    batch_size=batch_size,
                    temperature=temperature,
                    opt=optimizer)
            # train_mse.update_state(train_loss)
            step_wise_train_loss.append(train_loss)
        epoch_wise_train_loss.append(np.mean(step_wise_train_loss))

        # for valid_batch in valid_dataset:
        #     mse = valid_step(model, valid_batch)
        

        x1_val = pretraining_records(os.path.join(path, 'val'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.0,
                                        rnd_frac=0.0,
                                        same_frac=0.0)

        x2_val = pretraining_records(os.path.join(path, 'val'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.0,
                                        rnd_frac=0.0,
                                        same_frac=0.0)

        x3_val = pretraining_records(os.path.join(path, 'val'),
                                        batch_size,
                                        max_obs=maxlen,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=0.0,
                                        rnd_frac=0.0,
                                        same_frac=0.0)
        
        for i_val, j_val, k_val in zip(x1_val, x2_val, x3_val):

            # i_val = add_white_noise(i_val, redshift=redshift)
            # j_val = add_white_noise(j_val, redshift=redshift)
            # k_val = add_white_noise(k_val, redshift=redshift)

            val_loss = train_step(model, i_val, j_val, k_val , 
                    batch_size=batch_size,
                    temperature=temperature,
                    opt=optimizer)
            step_wise_val_loss.append(val_loss)
            # valid_mse.update_state(val_loss)
        
        epoch_wise_val_loss.append(np.mean(step_wise_val_loss))

        msg = 'EPOCH {} - ES COUNT: {}/{} train xbent: {:.5f} - val xbent: {:.5f}'.format(epoch,
                                                                                      es_count,
                                                                                      patience,
                                                                                      np.mean(step_wise_train_loss),
                                                                                      np.mean(step_wise_val_loss))

        pbar.set_description(msg)

        save_scalar(train_writter, np.mean(step_wise_train_loss), epoch, name='xbent')
        save_scalar(valid_writter, np.mean(step_wise_val_loss), epoch, name='xbent')

        mean_val_loss = np.mean(step_wise_val_loss)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            es_count = 0.
            model.save_weights(os.path.join(exp_path, 'weights.h5'))
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break
        
        # train_mse.reset_states()
        # valid_mse.reset_states()
        train_loss = 0
        val_loss = 0
    with open(f"{exp_path}/logs/train_loss.pickle", "wb") as output_file:
        pickle.dump(epoch_wise_train_loss, output_file)
    with open(f"{exp_path}/logs/val_loss.pickle", "wb") as output_file:
        pickle.dump(epoch_wise_val_loss, output_file)
        


def predict(model,
            dataset,
            conf,
            predic_proba=False):

    total_mse, inputs, reconstructions = [], [], []
    masks, times = [], []
    for step, batch in tqdm(enumerate(dataset), desc='prediction'):
        mse, x_pred, x_true = valid_step(model,
                                         batch,
                                         return_pred=True,
                                         normed=True)

        total_mse.append(mse)
        times.append(batch['times'])
        inputs.append(x_true)
        reconstructions.append(x_pred)
        masks.append(batch['mask_out'])

    res = {'mse':tf.reduce_mean(total_mse).numpy(),
           'x_pred': tf.concat(reconstructions, 0),
           'x_true': tf.concat(inputs, 0),
           'mask': tf.concat(masks, 0),
           'time': tf.concat(times, 0)}

    return res
