import tensorflow as tf
import argparse
import logging
import json
import time
import os
import pickle
import numpy as np
from core.astromer import get_ASTROMER, train
from core.data  import pretraining_records
from core.utils import get_folder_name
from time import gmtime, strftime

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

from google.protobuf.json_format import MessageToJson

def run1(opt):

    test_batches = pretraining_records(os.path.join(opt.data, 'val'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=opt.repeat,
                                        msk_frac=opt.msk_frac,
                                        rnd_frac=opt.rnd_frac,
                                        same_frac=opt.same_frac)
    raw_dataset = tf.data.TFRecordDataset("main-code/data/sim-ZTF-data_cuts/records/ZTF/r-LCs/val/Ia/chunk_0.record")

    # for raw_record in raw_dataset:
    #     # print(raw_record)
    #     example = tf.train.SequenceExample()
    #     example.ParseFromString(raw_record.numpy())
    #     # m = json.loads(MessageToJson(example))
    #     # print(m['features']['feature'].keys())
    #     print(example)
    #     break
    print(test_batches)


    # for i, j, k in zip(test_batches, test_batches, test_batches):
    #         # print(type(i), type(j), type(k))
    #         print(type(i["input"]), i["length"].numpy()[0])
    #         len_ = i["length"].numpy()[0]
    #         masked_flux = i["input"].numpy()
    #         flux = masked_flux[:, :(len_), :]
    #         # noise_scale = np.random.lognormal(0, 0.50)
    #         # noise_sigmas = np.random.lognormal(np.log(noise_scale), 1., len_)
    #         # # Add the noise to the observations.
    #         # noise = np.random.normal(0., noise_sigmas)
    #         noise = np.random.normal(0, 20, len_)
    #         noise = noise.reshape((1, len_, 1))
    #         print(noise.shape)
    #         flux_ = flux + noise
    #         print(masked_flux)
    #         masked_flux[:, :(len_), :] = flux_
    #         tensor = tf.convert_to_tensor(masked_flux)
            
    #         i["input"] = tensor
    #         print(type(masked_flux), type(i["input"]), type(tensor))
            
    #         break
    # majumder/astromer/main-code/runs/ZTF/logs
    # with open(f"main-code/runs/ztf/logs/train_loss.pickle", "rb") as output_file:
    #     train=pickle.load(output_file)
    # with open(f"main-code/runs/ztf/logs/val_loss.pickle", "rb") as output_file:
    #     val=pickle.load(output_file)

    # print(train, val)


def run(opt):
    
    path_to_store="main-code/enc-astromer/ztf/r-band/astro_embeddings"

    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    # Check for pretrained weigths
    if os.path.isfile(os.path.join(opt.w, 'weights.h5')):
        os.makedirs(opt.p, exist_ok=True)


        print('[INFO] Pretrained model detected! - Finetuning...')
        conf_file = os.path.join(opt.w, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        # Loading hyperparameters of the pretrained model
        astromer = get_ASTROMER(num_layers=conf['layers'],
                                d_model=conf['head_dim'],
                                meta_shape=conf['meta_shape'],
                                emd_dim=conf['emd_dim'],
                                num_heads=conf['heads'],
                                dff=conf['dff'],
                                base=conf['base'],
                                dropout=conf['dropout'],
                                maxlen=conf['max_obs'],
                                use_leak=conf['use_leak'],
                                no_train=conf['no_train'])


        test_batches = pretraining_records(os.path.join(opt.data, 'test'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=opt.repeat,
                                        msk_frac=opt.msk_frac,
                                        rnd_frac=opt.rnd_frac,
                                        same_frac=opt.same_frac)

        # Loading pretrained weights
        weights_path = '{}/weights.h5'.format(opt.w)
        
        astromer.load_weights(weights_path)

        astromer.trainable = False
        encoder = astromer.get_layer('encoder')
        #
        n_samples = list(test_batches)
        embeddings=np.zeros((len(n_samples), conf['head_dim']))
        labels=np.zeros((len(n_samples),), dtype=int)
        
        for i, batch in enumerate(test_batches):
            print("\ni:", i)
            emb = encoder(batch)
            emb = emb._numpy()
            # print(emb.shape)
            # break
            # mean_emb = np.array(np.nanmean(emb, 1))
            mean_emb = tf.math.reduce_mean(emb, axis=1)
            embeddings[i]=mean_emb
            labels[i]=batch["label"]._numpy()[0]
        np.savez_compressed(f"{path_to_store}", a=embeddings, b=labels)
        
    else:
        print('[ERROR] No weights found to load')
    

if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser()
    #
    # DATA
    #
    parser.add_argument('--max-obs', default=20, type=int,
                    help='Max number of observations')
    parser.add_argument('--meta-shape', default=54, type=int,
                    help='Number of light-curve features used as metadata')
    parser.add_argument('--emd-dim', default=10, type=int,
                    help='embeddings shape of the light curves/input to the transformer')
    parser.add_argument('--repeat', default=1, type=int,
                    help='Number of times for sampling windows from single LC')
    parser.add_argument('--msk-frac', default=0.0, type=float,
                        help='[MASKED] fraction')
    parser.add_argument('--rnd-frac', default=0.0, type=float,
                        help='Fraction of [MASKED] to be replaced by random values')
    parser.add_argument('--same-frac', default=0.0, type=float,
                        help='Fraction of [MASKED] to be replaced by same values')
    #
    # TRAINING PAREMETERS
    #
    # majumder/simclr/main-code/data/sim-ZTF-data_cuts-newfeatures/records/ZTF/r-LCs/test
    parser.add_argument('--data', default='main-code/data/sim-ZTF-data_cuts-newfeatures/records/ZTF/r-LCs/', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="main-code/runs/ztf/", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--w', default="main-code/runs/ztf/", type=str,
                        help='astromer weigths')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--gpu', default='5', type=str,
                        help='GPU to use')
    opt = parser.parse_args()
    #
    run(opt)
    #
    end = time.time()
    duration = end-start
    #
    print("Time (mins): ", duration//60)
    print("Done!")