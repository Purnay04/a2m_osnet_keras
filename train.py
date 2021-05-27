import os
import sys
import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from absl import app, logging, flags
from absl.flags import FLAGS

from dataset_manager import Market1501
from models.osnet import OSNet
from losses import CrossEntropyLabelSmooth, DeepSupervision
from models.models import get_optimizer
# Importing dependencies

# adding cmd args
# optimization Options 
flags.DEFINE_string('optim', 'sgd', 'optimization algorithm')
flags.DEFINE_integer('max_epoch', 350, 'maximum epoch to run')
flags.DEFINE_integer('start_epoch', 0, 'manual epoch number')
flags.DEFINE_integer('train_batch', 64, 'train batch size')
flags.DEFINE_integer('test_batch', 32, 'test batch size')
flags.DEFINE_float('lr', 0.065, 'initial learning rate')
flags.DEFINE_integer('stepsize', 20, 'stepsize to decay learning rate')
flags.DEFINE_float('gamma', 0.1, 'learning rate decay')
#flags.DEFINE_enum('lr_sched', 'linear', ['linear', 'exp'], 'linear : linear Way' 'exp : Exponetial Way')
flags.DEFINE_float('weight_decay', 5e-04, 'weight decay')

# Miscs
flags.DEFINE_integer('print_freq', 10, 'print frequency') #10
flags.DEFINE_integer('seed', 1, 'manual seed')
flags.DEFINE_string('resume', '', 'PATH')
flags.DEFINE_bool('evaluate', False, 'evaluation only')
flags.DEFINE_integer('eval_step', 1, 'run evaluation for every N epochs')
flags.DEFINE_integer('start_eval', 0, 'start to evaluate after specific epoch')
flags.DEFINE_string('save_dir', 'logs', 'save directory')
flags.DEFINE_bool('use_cpu', False, 'evaluation only')
flags.DEFINE_string('gpu_devices', '0', 'gpu device ids for CUDA_VISIBLE_DEVICES')


train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            featurewise_std_normalization=True,
            # translation is remaining
            )
test_datagen = ImageDataGenerator(
            featurewise_std_normalization=True,
            # translation is remaining
        )
def transform(data, test=False):
    imgs_path, rest = data[:,0], data[:,1:].astype(np.int32)
    imgs = np.array(list(map(lambda x: img_to_array(load_img(x)), imgs_path)))
    if not test:
        train_datagen.fit(imgs)
        return train_datagen, imgs, rest
    else:
        test_datagen.fit(imgs)
        return test_datagen, imgs, rest
    
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis = 1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # remove gallery sample that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identify does not appear in gallery
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP
    
def main(_argv):
    # setting up cuda visibility
    tf.random.set_seed(FLAGS.seed)
    if not FLAGS.use_cpu:
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_devices
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)    
        use_gpu = True
    else:
        use_gpu = False
    
    now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    # gather and manage dataset from dataset manager
    print("Initializing dataset")
    dataset = Market1501()
    print("Dataset initialized")
    #print(dataset.train[0])
    # create data loader for train, query & gallery
    print("processing dataset")
    train_datagen, train_imgs, train_labels = transform(np.array(dataset.train))
    print("train done")
    query_datagen, query_imgs, query_labels = transform(np.array(dataset.query), test=True)
    print("query done")
    gallery_datagen, gallery_imgs, gallery_labels = transform(np.array(dataset.gallery), test=True)
    print("gallery done")
    
    # creating model, criterion, optimizer, stepLR if it's present
    model = OSNet(classes=dataset.num_train_pids, loss_type="triplet")
    raw_output = model(Input(shape=(256, 128, 3)), training=True)
    
    #total_para = np.sum([K.count_params(w) for w in model.trainable_weights]) + np.sum([K.count_params(w) for w in model.non_trainable_weights])
    print(model.summary())
    """
    imgs, labels = train_datagen.flow(train_imgs, train_labels, batch_size=FLAGS.train_batch)[0]
    print(imgs.shape, labels.shape)
    """

   
    criterion = CrossEntropyLabelSmooth(num_classes= dataset.num_train_pids)
    optimizer = get_optimizer(FLAGS.optim, FLAGS.lr, FLAGS.weight_decay)
    lr_decay_epoches = [150, 225, 300]
    start_epoch = FLAGS.start_epoch
    # conditional branch for resuming_training, evaluate
    if FLAGS.resume != '':
        print("Loading checkpoints from {}".format(FLAGS.resume))
        model.load_weights(FLAGS.resume)
        name = FLAGS.resume.split("/")[-1]
        start_epoch = int(name.split("_")[2].replace("epoch", ""))
        now += "_resumed"
        #print(start_epoch)
    summary_writer = tf.summary.create_file_writer('logs/{}'.format(now))
    # initialize start_time, trian_time, best_rank1, best_epoch
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    
    
    # iterate epoch
    for epoch in range(start_epoch, FLAGS.max_epoch):
        print("\tCurrent epoch:{}".format(epoch+1))
        tf.summary.trace_on(graph=True)
        # initialize train_time
        start_train_time = time.time()
        # make 1 epoch with training set / train
        losses, accuracy = train(epoch, model, criterion, optimizer, train_datagen, train_imgs, train_labels)
        train_time += round(time.time() - start_train_time)
        # calculate train_time

        # make scheduler steps as per epoch condition
        if FLAGS.stepsize > 0 and epoch+1 in lr_decay_epoches:
            optimizer.lr = optimizer.lr - (optimizer.lr * 0.1)
            
        # make evaluation/validation by checking current epoch+1 is multiple of 10
        if (epoch+1) % 10 == 0:                      # 10
            print("==> Test")
            rank1, mAP = test(model, query_datagen, query_imgs, query_labels, gallery_datagen, gallery_imgs, gallery_labels)
            
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            
            with summary_writer.as_default():
                tf.summary.scalar('rank-1', rank1, step=epoch+1)
                tf.summary.scalar('mAP', mAP, step=epoch+1)
            model.save_weights(
                'checkpoints/{}/osnet_train_epoch{}_rank1{}.tf'.format(now, epoch, rank1))
            
        with summary_writer.as_default():
            tf.summary.scalar('loss', losses.result(), step=epoch+1)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch+1)
            tf.summary.scalar('lr:{}'.format(FLAGS.optim), optimizer.lr, step=epoch+1)
            tf.summary.trace_export('name', step=0)

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
            # make batch prediction with gallery set and query / make test
            # check rank1 if it is best then change best_rank1 and epoch otherwise leave
            
            # saves model parameter
    # print out result with best rank and epoch
    # print total time for training 

def train(epoch, model, criterion, optimizer, train_datagen, train_imgs, train_labels):
    """
    # creating losses_paras, batch_time_paras, data_time_paras with AverageMeter
    # make train the model
    # total end time
    
    # iterate as per batches
        # converts images and pids from cuda
        
        # update data_time with current time
        
        # make batch prediction with train_images
        # calculate loss
        # optimizer zero grad
        # backpropogate loss
        # optimizer step
        
        # update batch_time, end, losses
        # print epoch status with epoch val, learning rate, time, data, loss
    Args:
        epoch ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        train_datagen ([type]): [description]
        train_imgs ([type]): [description]
        train_labels ([type]): [description]
    """
    accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)
    losses = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    batch_time = tf.keras.metrics.Mean('train_batch_time', dtype=tf.float32)
    data_time = tf.keras.metrics.Mean('data_time', dtype=tf.float32)
    
    end = time.time()
    trian_max_batches = train_imgs.shape[0] // FLAGS.test_batch if train_imgs.shape[0] % FLAGS.train_batch == 0 else (train_imgs.shape[0] // FLAGS.train_batch) + 1
    for batch_idx, (imgs, labels) in enumerate(train_datagen.flow(train_imgs, train_labels, batch_size=FLAGS.train_batch)):
        #print("batch id:{}".format(batch_idx))
        pids, camids = labels[:,0], labels[:,1]
        data_time.update_state(time.time() - end)
        
        with tf.GradientTape() as tape:
            outputs = model(imgs, training=True)
            #print(outputs)
            
            if isinstance(outputs, tuple):
                loss = DeepSupervision(criterion, outputs, pids)
            else:
                loss = criterion(outputs, pids)
            #print("loss:",loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))
        
        batch_time.update_state(time.time() - end)
        end = time.time()
        
        #print("total loss:",loss * pids.shape[0])
        if isinstance(outputs, tuple):
            targets = tf.tensor_scatter_nd_update(tf.zeros(outputs[0].shape), tf.transpose([tf.range(outputs[0].shape[0]), pids]), tf.ones(outputs[0].shape[0]))
            accuracy.update_state(targets, outputs[0])
        else:
            targets = tf.tensor_scatter_nd_update(tf.zeros(outputs.shape), tf.transpose([tf.range(outputs.shape[0]), pids]), tf.ones(outputs.shape[0]))
            accuracy.update_state(targets, outputs)
        losses.update_state(loss * pids.shape[0])
        temp_lr = optimizer.lr.numpy()
        
        if (batch_idx+1) % FLAGS.print_freq == 0:
            print('\tEpoch: [EP NO: {0}][BT NO: {1}]\t'
                  'LearningRate {temp_lr:.4f}\t'
                  'Time ({batch_time:.3f})\t'
                  'Data ({data_time:.3f})\t'
                  'Loss ({loss:.4f})\t'.format(
                   epoch+1, batch_idx+1, temp_lr=temp_lr, batch_time=batch_time.result().numpy(),
                   data_time=data_time.result().numpy(), loss=losses.result().numpy()))
            
        if batch_idx+1 == trian_max_batches:          #trian_max_batches
            break 
    return losses, accuracy

def test(model, query_datagen, query_imgs, query_labels, gallery_datagen, gallery_imgs, gallery_labels, ranks=[1, 5, 10, 20]):
    """
    # creating batch_time with AverageMeter
    # make eval the model
    
    # with no_grad
        # create query_features, query_ids, query_camids
        # iterate as per query batches
            # convert images & set end time
            # make prediction and update batch_time
            
            # update query_features, query_ids, query_camids
        # convery query_features, query_ids, query_camids into array
        
        # create gallery_features, gallery_ids, gallery_camids
        # iterate as per gallery batches
            # convert images & set end time
            # make prediction and update batch_time
            
            # update gallery_features, gallery_ids, gallery_camids
        # convery gallery_features, gallery_ids, gallery_camids into array
    
    # creates m, n where m for total query_features & n for total gallery_features
    # calculate dismat
    # convert dismat to numpy array
    
    # calculate cmc, mAP from evaluate function
    # print out mAP and CMC curve

    Args:
        model ([type]): [description]
        query_datagen ([type]): [description]
        query_imgs ([type]): [description]
        query_labels ([type]): [description]
        gallery_datagen ([type]): [description]
        gallery_imgs ([type]): [description]
        gallery_labels ([type]): [description]
        ranks (list, optional): [description]. Defaults to [1, 5, 10, 20].
    """
    batch_time = tf.keras.metrics.Mean('test_batch_time', dtype=tf.float32)
    
    query_max_batches = query_imgs.shape[0] // FLAGS.test_batch if query_imgs.shape[0] % FLAGS.test_batch == 0 else (query_imgs.shape[0] // FLAGS.test_batch) + 1
    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, labels) in enumerate(query_datagen.flow(query_imgs, query_labels, batch_size=FLAGS.test_batch)):
        pids, camids = labels[:,0], labels[:,1]
        end = time.time()
        
        features = model(imgs)
        batch_time.update_state(time.time() - end)
        
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
        
        if batch_idx + 1 == query_max_batches:
            break
    qf = tf.concat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    
    print("\tExtracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))
    
    gallery_max_batches = gallery_imgs.shape[0] // FLAGS.test_batch if gallery_imgs.shape[0] % FLAGS.test_batch == 0 else (gallery_imgs.shape[0] // FLAGS.test_batch) + 1
    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, labels) in enumerate(gallery_datagen.flow(gallery_imgs, gallery_labels, batch_size=FLAGS.test_batch)):
        pids, camids = labels[:,0], labels[:,1]
        end = time.time()
        
        features = model(imgs)
        batch_time.update_state(time.time() - end)
        
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
        
        if batch_idx + 1 == gallery_max_batches:
            break
    gf = tf.concat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    
    print("\tExtracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.result().numpy(), FLAGS.test_batch))
    

    m, n = qf.shape[0], gf.shape[0]
    distmat = tf.broadcast_to(tf.reduce_sum(tf.pow(qf, 2), axis=1, keepdims=True), [m, n]) + \
              tf.transpose(tf.broadcast_to(tf.reduce_sum(tf.pow(gf, 2), axis=1, keepdims=True), [n, m]))
    distmat = distmat + (tf.matmul(qf, tf.transpose(gf)) * -2)
    distmat = distmat.numpy()
    
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print("Results ----------")
    print("\tmAP: {:.1%}".format(mAP))
    print("\tCMC curve")
    for r in ranks:
        print("\tRank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("\t------------------")
    
    return cmc[0], mAP
    


if __name__ == "__main__":
    app.run(main)