#### import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import pandas as pd
import os 
tf.compat.v1.enable_v2_behavior
# Import tensornetwork
import tensornetwork as tn
# Set the backend to tesorflow
# (default is numpy)
tn.set_default_backend("tensorflow")
batch_size = 64
learning_rate = 0.0001
class ClutteredMNISTDataset(Sequence):
    reg_dataset_size = 11276

    def __init__(self, base_path, csv_path, data_scaling=1., num_examples=None, balance=None, num_classes=2):
        self.base_path = base_path
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        self.data_scaling = data_scaling
        self.num_examples = num_examples
        self.balance = balance
        self.num_classes = num_classes
        self.img_paths = self.csv['img_path'].values
        self.lbls = self.csv['label'].values.astype(np.int32)
        self.weights = np.ones([len(self.img_paths), ])

        if self.num_examples is not None and self.balance is not None:
            # only rebalance if examples and balance is given
            assert self.num_examples <= len(self.img_paths), \
                'not enough examples in dataset {} - {}'.format(self.num_examples, len(self.img_paths))

            pos_num = int(self.balance * self.num_examples)
            neg_num = self.num_examples - pos_num

            pos_mask = (self.lbls == 1)
            pos_paths = self.img_paths[pos_mask][:pos_num]

            neg_mask = (self.lbls == 0)
            neg_paths = self.img_paths[neg_mask][:neg_num]

            self.img_paths = np.concatenate([pos_paths, neg_paths], 0)
            self.lbls = np.concatenate([np.ones([pos_num, ]), np.zeros([neg_num, ])], 0)
            self.weights = np.ones([self.num_examples, ])
            self.weights[:pos_num] /= pos_num
            self.weights[pos_num:] /= neg_num

        self.shrinkage = self.reg_dataset_size // len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = os.path.join(self.base_path, self.img_paths[index])
        lbl = self.lbls[index].astype(np.int64)
        lbl = np.eye(self.num_classes)[lbl].astype(np.int64)
        img = np.load(path)
        if isinstance(img, np.lib.npyio.NpzFile):
            img = img['arr_0']

        if self.data_scaling != 1.:
            img = zoom(img, self.data_scaling)
            img = img.clip(0., 1.)

        img = (img[:,:,np.newaxis].astype(np.float32).astype(np.float32) -0.5)/1

        return img, lbl
    

class Batcher(Sequence):
    """Assemble a sequence of things into a sequence of batches."""
    def __init__(self, sequence, batch_size=16):
        self._batch_size = batch_size
        self._sequence = sequence
        self._idxs = np.arange(len(self._sequence))

    def __len__(self):
        return int(np.ceil(len(self._sequence) / self._batch_size))

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Index out of bounds")

        start = i*self._batch_size
        end = min(len(self._sequence), start+self._batch_size)
        data = [self._sequence[j] for j in self._idxs[start:end]]
        inputs = [d[0] for d in data]
        outputs = [d[1] for d in data]

        return self._stack(inputs), self._stack(outputs)

    def _stack(self, data):
        if data is None:
            return None

        if not isinstance(data[0], (list, tuple)):
            return np.stack(data)

        seq = type(data[0])
        K = len(data[0])
        data = seq(
            np.stack([d[k] for d in data])
            for k in range(K)
        )

        return data

    def on_epoch_end(self):
        np.random.shuffle(self._idxs)
        self._sequence.on_epoch_end()
        
        
        
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_v2_behavior
# Import tensornetwork
import tensornetwork as tn
from tensorflow.keras.utils import to_categorical
tn.set_default_backend("tensorflow")

data_path = '/datacommons/carin/fk43/NeedleinHaystack/mnist64_0/'
train_data=ClutteredMNISTDataset(base_path=data_path, csv_path=os.path.join(data_path, 'train.csv'), data_scaling=1., num_examples=3000, balance=0.5)
val_data = ClutteredMNISTDataset(base_path=data_path, csv_path=os.path.join(data_path, 'val.csv'), data_scaling=1., num_examples=1000, balance=0.5)
test_data = ClutteredMNISTDataset(base_path=data_path, csv_path=os.path.join(data_path, 'test.csv'), data_scaling=1., num_examples=1000, balance=0.5)


train_data = Batcher(train_data, batch_size=batch_size)
val_data = Batcher(val_data, batch_size=batch_size)
test_data = Batcher(test_data, batch_size=batch_size)



class GridMERAin16(tf.keras.layers.Layer):
    
    def __init__(self, kernel_dims, bond_dims, output_dims):
        super(GridMERAin16, self).__init__()
        # Create the variables for the layer.
        # In this case, the input tensor is (, 1936), we factorize it into a tensor (, 11, 11, 16)
        # first_dim: output shape?
        # second_dim: connect with data tensor
        # third_dim: inter-connect
        in_dims = int((kernel_dims//4)**2)
        self.entanglers = []
        self.isometries= []
        self.kernel_dims = kernel_dims
        self.output_dims = output_dims
        #entanglers
        self.entanglers1 = tf.Variable(tf.random.normal
                                             (shape=(in_dims, in_dims, 
                                                     in_dims, in_dims, bond_dims, bond_dims, bond_dims, bond_dims),
                                              stddev=1.0/10000), 
                                              trainable=True)
        self.entanglers2 = tf.Variable(tf.random.normal
                                             (shape=(bond_dims, bond_dims, 
                                                     bond_dims, bond_dims, bond_dims, bond_dims, bond_dims, bond_dims),
                                              stddev=1.0/10000), 
                                              trainable=True)
        # isometries
        self.isometries1 = [tf.Variable(tf.random.normal(shape=(in_dims, in_dims, in_dims, 
                                                                            bond_dims, bond_dims)
                                                                     , stddev=1.0/100000),
                                            trainable=True), 
                           tf.Variable(tf.random.normal(shape=(in_dims, in_dims, bond_dims, 
                                                                            in_dims, bond_dims)
                                                                     , stddev=1.0/100000),
                                            trainable=True),
                           tf.Variable(tf.random.normal(shape=(in_dims, bond_dims, in_dims, 
                                                                            in_dims, bond_dims)
                                                                     , stddev=1.0/100000),
                                            trainable=True),
                           tf.Variable(tf.random.normal(shape=(bond_dims, in_dims, in_dims, 
                                                                            in_dims, bond_dims)
                                                                     , stddev=1.0/100000),
                                            trainable=True)]
        
        self.isometries2 = tf.Variable(tf.random.normal(shape=(bond_dims, bond_dims, bond_dims, 
                                                                            bond_dims, output_dims)
                                                                     , stddev=1.0/100000),
                                            trainable=True)

        #print(self.final_mps.shape)
        self.bias = tf.Variable(tf.zeros(shape=(output_dims,)), name="bias", trainable=True)


    def call(self, inputs):
        # Define the contraction.
        # We break it out so we can parallelize a batch using tf.vectorized_map.
        def f(input_vec, entanglers1, entanglers2, isometries1, isometries2, bias_var, kernel_dims):
            input_vv = []
            step = int(kernel_dims//4)
            for i in range(4):
                for ii in range(4):
                    input_vv.append(tf.reshape(input_vec[i*step:i*step+step, ii*step:ii*step+step, 0], (1, step**2)))
            input_vec = tf.concat(input_vv, axis=0)
            input_vec = tf.reshape(input_vec, (16, step**2))
            input_vec = tf.unstack(input_vec)
            input_nodes = []
            for e_iv in input_vec:
                input_nodes.append(tn.Node(e_iv))
            
            e_nodes1 = tn.Node(entanglers1)
            e_nodes2 = tn.Node(entanglers2)
                
                                     
            isometries_nodes1 = []
            for eiso in isometries1:
                isometries_nodes1.append(tn.Node(eiso))
            isometries_nodes2 = tn.Node(isometries2)
            
            
            e_nodes1[0] ^ input_nodes[5][0]
            e_nodes1[1] ^ input_nodes[6][0]
            e_nodes1[2] ^ input_nodes[9][0]
            e_nodes1[3] ^ input_nodes[10][0]

            e_nodes1[4] ^ isometries_nodes1[0][3]
            e_nodes1[5] ^ isometries_nodes1[1][2]
            e_nodes1[6] ^ isometries_nodes1[2][1]
            e_nodes1[7] ^ isometries_nodes1[3][0]     
            
            input_nodes[0][0] ^ isometries_nodes1[0][0]
            input_nodes[1][0] ^ isometries_nodes1[0][1]
            input_nodes[4][0] ^ isometries_nodes1[0][2]
            
            input_nodes[2][0] ^ isometries_nodes1[1][0]
            input_nodes[3][0] ^ isometries_nodes1[1][1]
            input_nodes[7][0] ^ isometries_nodes1[1][3]
            
            input_nodes[8][0] ^ isometries_nodes1[2][0]
            input_nodes[12][0] ^ isometries_nodes1[2][2]
            input_nodes[13][0] ^ isometries_nodes1[2][3]
            
            input_nodes[11][0] ^ isometries_nodes1[3][1]
            input_nodes[14][0] ^ isometries_nodes1[3][2]
            input_nodes[15][0] ^ isometries_nodes1[3][3]
            
            
            isometries_nodes1[0][4] ^ e_nodes2[0]
            isometries_nodes1[1][4] ^ e_nodes2[1]
            isometries_nodes1[2][4] ^ e_nodes2[2]
            isometries_nodes1[3][4] ^ e_nodes2[3]

            e_nodes2[4] ^ isometries_nodes2[0]
            e_nodes2[5] ^ isometries_nodes2[1]
            e_nodes2[6] ^ isometries_nodes2[2]
            e_nodes2[7] ^ isometries_nodes2[3]

                            
            nodes = tn.reachable(isometries_nodes2)
            result = tn.contractors.greedy(nodes)
            result = result.tensor
            return result + bias_var

        # To deal with a batch of items, we can use the tf.vectorized_map function.
        # https://www.tensorflow.org/api_docs/python/tf/vectorized_map
        output = tf.vectorized_map(lambda vec: f(vec, self.entanglers1, self.entanglers2,
                                                 self.isometries1,  self.isometries2, self.bias, self.kernel_dims), inputs)
        return tf.reshape(output, (-1, self.output_dims))
    
from tensorflow.keras.layers import Lambda, Input, Concatenate, Reshape, Softmax, Dense, Flatten
from tensorflow.keras.models import Model, Sequential

def get_model(input_shape=(64, 64, 1)):
    x_in = Input(shape=input_shape)
    x_out_list = []
    for i in range(8):
        for j in range(8):
            subx = Lambda(lambda x:x[:, 8*i:8*(i+1),8*j:8*(j+1),:] )(x_in)
            x_out_list.append(GridMERAin16(kernel_dims=8, bond_dims=2, output_dims=1)(subx))
    x_out = Concatenate(axis=1)(x_out_list)
    x_out = Reshape(target_shape=(8, 8, 1))(x_out)
    y = GridMERAin16(kernel_dims=8, bond_dims=2, output_dims=2)(x_out)
    y = Softmax()(y)
    return Model(inputs=x_in, outputs=y)

def get_dense_model(input_shape=(64, 64, 1)):
    x_in = Input(shape=input_shape)
    x_out_list = []
    for i in range(4):
        for j in range(4):
            subx = Lambda(lambda x:x[:, 16*i:16*(i+1),16*j:16*(j+1),:] )(x_in)
            x_out_list.append(GridMERAin16(kernel_dims=16, bond_dims=2, output_dims=2)(subx))
    x_out = Concatenate(axis=1)(x_out_list)
    x_out = Flatten()(x_out)
    y = Dense(2, activation='softmax')(x_out)
    #x_out = Reshape(target_shape=(8, 8, 1))(x_out)
    #y = GridMERAin16(kernel_dims=8, bond_dims=2, output_dims=2)(x_out)
    #y = Softmax()(y)
    return Model(inputs=x_in, outputs=y)

def get_loop_model(input_shape=(64, 64, 1)):
    x_in = Input(shape=input_shape)
    x_out_list = []
    MERA_layer = GridMERAin16(kernel_dims=16, bond_dims=2, output_dims=2)
    for i in range(4):
        for j in range(4):
            subx = Lambda(lambda x:x[:, 16*i:16*(i+1),16*j:16*(j+1),:] )(x_in)
            x_out_list.append(MERA_layer(subx))
    x_out = Concatenate(axis=1)(x_out_list)
    x_out = Flatten()(x_out)
    y = Dense(2, activation='softmax')(x_out)
    #x_out = Reshape(target_shape=(8, 8, 1))(x_out)
    #y = GridMERAin16(kernel_dims=8, bond_dims=2, output_dims=2)(x_out)
    #y = Softmax()(y)
    return Model(inputs=x_in, outputs=y)

def get_deep_model(input_shape=(64, 64, 1)):
    x_in = Input(shape=input_shape)
    x_in2_list = []
    for i in range(16):
        for j in range(16):
            subx = Lambda(lambda x:x[:, 4*i:4*(i+1),4*j:4*(j+1),:] )(x_in)
            x_in2_list.append(GridMERAin16(kernel_dims=1, bond_dims=2, output_dims=1)(subx))
    x_in2 = Concatenate(axis=1)(x_in2_list)
    x_in2 = Reshape(target_shape=(16, 16, 1))(x_in2)
    x_out_list = []
    for i in range(4):
        for j in range(4):
            subx = Lambda(lambda x:x[:, 4*i:4*(i+1),4*j:4*(j+1),:] )(x_in2)
            x_out_list.append(GridMERAin16(kernel_dims=1, bond_dims=2, output_dims=1)(subx))
    x_out = Concatenate(axis=1)(x_out_list)
    x_out = Reshape(target_shape=(4, 4, 1))(x_out)
    y = GridMERAin16(kernel_dims=1, bond_dims=2, output_dims=2)(x_out)
    y = Softmax()(y)
    return Model(inputs=x_in, outputs=y)


MERA_model_64 = get_dense_model(input_shape=(64, 64, 1))

MERA_model_64.summary()

# TensorNetwork model
MERA_model_64.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=0.005), metrics=['accuracy'])
MERA_model_64_hist = MERA_model_64.fit_generator(train_data, validation_data=val_data, epochs=50, verbose=1)

# TN model
MERA_model_64.evaluate_generator(test_data, int(np.ceil(1000/batch_size)), workers = 1)

tn_loss = MERA_model_64_hist.history['loss']
tn_acc = MERA_model_64_hist.history['accuracy']


np.savetxt('loss_MERA64.out', tn_loss, delimiter=',')  
np.savetxt('acc_MERA64.out', tn_acc, delimiter=',')  
