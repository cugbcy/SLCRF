import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import torch
import torch.utils.data
from scipy.io import loadmat
import scipy.optimize
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from skimage import morphology
from torch.autograd import Variable
import os
import math
import time
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import scipy.io

def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.kaiming_normal_(m.weight.data)

def reset_grad(*nets):
    for net in nets:
        net.zero_grad()

def save_mat(images, image_path):
    return scipy.io.savemat(image_path, {'array': images})

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import scipy.io
import os
import torch
import numpy as np
from skimage import morphology
import scipy.misc
from PIL import Image
from scipy.io import loadmat
import h5py
from sklearn.model_selection import train_test_split

import scipy.io
import numpy as np
from random import shuffle
import random
import scipy.ndimage
from skimage.util import pad
import os
import time
import math
from scipy.io import loadmat
from sklearn.decomposition import PCA
import h5py
# import matplotlib.pyplot as plt

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def per_class_acc(predictions, label_tensor):
    labels = label_tensor.cpu().data.numpy().flatten()
    num_class = predictions.shape[1]
    hist = np.zeros((num_class, num_class))
    hist = fast_hist(labels,predictions.cpu().data.numpy().argmax(1),num_class)
    hist = hist.astype(np.float64)
    acc_total = (np.diag(hist).sum() / hist.sum())
    Pee = 0.0
    for iii in range(num_class):
        Pee = Pee+hist.sum(1)[iii]*hist.sum(0)[iii]
    Pe= Pee / (hist.sum()*hist.sum())
    kappa = (acc_total - Pe) / (1 - Pe)
    AA = 0.0
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = (np.diag(hist)[ii] / float(hist.sum(1)[ii]))
            AA += acc
        print("    class # %d accuracy = %f " % (ii, (acc*100)))
    AA = AA / (num_class)
    print ('OA = %f' % (acc_total*100))
    print ('AA  = %f' % (AA*100))
    print ('kappa  = %f' % kappa)
    return acc_total*100,AA*100,kappa


def dataloader(dataset, batch_size, random_rate, train=True, split='train'):
    HSIdataset = loadHSI(dataset,random_rate,train=train)
    data_loader = DataLoader(HSIdataset, batch_size=batch_size, shuffle=True)
    return data_loader, HSIdataset.test_data, HSIdataset.test_labels,HSIdataset.feature, HSIdataset.labels

def data_obt(dataset):

    if dataset == 'Indian_pines':
        h5file_name = os.path.expanduser(r'D:\TensorFlow\197\SLCRF\3DCAE_HSI\model\trained_by_indian\indian_CAE_feature.h5')
        file = h5py.File(h5file_name, 'r')
        data = file['feature'].value
        feature = np.reshape(data, [data.shape[0], -1])
        labels = file['label'].value
        feature = feature[labels > 0, :]
        labels = labels[labels > 0] - 1
    return feature, labels

class loadHSI():
    def __init__(self, data = 'Salinas',random_rate=0.05,  train = True, transform=None):
        self.train = train
        self.transform = transform
        self.data = data
        self.feature, self.labels = data_obt(self.data)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.feature,self.labels, train_size=random_rate)
        self.feature = torch.from_numpy(self.feature)
        self.test_data = torch.from_numpy(self.test_data)
        self.test_labels = self.test_labels.reshape(self.test_labels.shape[0], 1)
        self.n_classes = len(np.unique(self.test_labels))
        self.test_labels = np.asarray(np.eye(self.n_classes)[self.test_labels.astype(int)], dtype='int64')
        self.test_labels = torch.from_numpy(self.test_labels).squeeze(1)

        if self.train:
            self.train_data  = torch.from_numpy(self.train_data)
            self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], 1)
            self.n_classes = len(np.unique(self.train_labels))
            self.train_labels = np.asarray(np.eye(self.n_classes)[self.train_labels.astype(int)], dtype='int64')
            self.train_labels = torch.from_numpy(self.train_labels).squeeze(1)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        return img, target

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset = 'Indian_pines'
choices=['Salinas', 'Indian_pines','PaviaU','Pavia']

if dataset == 'Indian_pines':
    img = loadmat(r'D:\TensorFlow\local\Data\Indian_pines\HSI\Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = loadmat(r'D:\TensorFlow\local\Data\Indian_pines\HSI\Indian_pines_gt.mat')['indian_pines_gt']

img = img.astype(float)
Band = img.shape[2]
for band in range(Band):
    img[:,:,band] = (img[:,:,band]-np.min(img[:,:,band]))/(np.max(img[:,:,band])-np.min(img[:,:,band]))

random_rate = 0.05
#scipy.io.savemat(os.path.join(save_dir, 'pavia_train.mat'), {'array': train_gt})
mb_size = 256 # Batch size
d_step = 5   # Number of discriminator training steps for each generator training step
iter_number = 100000

data_loader, X_test, X_test_label,X_all,X_all_labels = dataloader(dataset, mb_size, random_rate,train=True)
# Number of bands
c_dim = data_loader.dataset.n_classes
X_dim = data_loader.__iter__().__next__()[0].shape[1]
class_weights = torch.ones((c_dim)).cuda()

class SparseAutoEncoder(nn.Module):

    def __init__(self):
        super(SparseAutoEncoder, self).__init__()
        rand = np.random.RandomState(int(time.time()))
        visible_size = X_dim
        hidden_size = X_dim // 2

        self.AE_1_forward = nn.Sequential(
            torch.nn.Linear(X_dim, hidden_size),
            nn.ReLU(),
        )
        self.AE_1_backward = nn.Sequential(
            torch.nn.Linear(hidden_size, X_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            torch.nn.Linear(hidden_size, c_dim),
        )
        self.apply(weight_init)

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def forward(self, x):
        h1 = self.AE_1_forward(x)
        x_reconstruct = self.AE_1_backward(h1)
        c = self.classifier(h1)
        return c,h1,x_reconstruct

mb_size = 256 # Batch size

# Get networks
C = SparseAutoEncoder().cuda()
criterionL2 = nn.MSELoss().cuda()
learningrate = 0.0001   # Learning rate 5e-5
optimizer = torch.optim.Adam(C.parameters(), lr=learningrate)
train_hist = {}
train_hist['C_loss'] = []
train_hist['R_loss'] = []
best_acc = 0.0
lamda2 = para.lamda2

for it in range(iter_number):

    C.train()
    for _, (X, y) in zip(range(5), data_loader):

        X, y = X.float(), y.float()
        X, y = X.cuda(), y.cuda()
        _, classes = torch.max(y, dim=1)

        pred, hidden, X_reconstruct = C(X)
        C_loss = F.cross_entropy(pred, classes, weight=class_weights)
        R_loss = criterionL2(X_reconstruct, X)
        loss = C_loss * lamda2 + R_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_hist['C_loss'].append(C_loss.item())

    #Print and plot every now and then
    if it % 1000 == 0:
        with torch.no_grad():
            print('Iter-{}; loss: {}; C_loss: {}'.format(it,
                                                                       loss.data.cpu().numpy(),
                                                                       C_loss.data.cpu().numpy(),
                                                                       ))
            X_test, X_label, X_all = X_test.float(), X_test_label.float(), X_all.float()
            X_test, X_label, z, X_all = X_test.cuda(), X_label.cuda(), z.cuda(), X_all.cuda()
            classes = torch.max(X_label, dim=1)[1]
            pred_ = C(X_test)[0]
            C_loss = F.cross_entropy(pred_, classes, weight=class_weights)
            print("test_results ",'C_loss: {}'.format(C_loss.data.cpu().numpy()))
            OA, AA, kappa = per_class_acc(pred_, classes)
            save_dir = './result/' + dataset
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if OA > best_acc and OA > 0:
                best_acc = OA
                pred_all = C(X_all)[0]
                fea_all = C(X_all)[1]
                All_reconstruct = C(X_all)[2]
                Fea_W1 = C.state_dict()['AE_1_forward.0.weight'].cpu().numpy()
                Fea_W2 = C.state_dict()['AE_1_backward.0.weight'].cpu().numpy()
                Fea_b1 = C.state_dict()['AE_1_forward.0.bias'].cpu().numpy()
                Fea_b2 = C.state_dict()['AE_1_backward.0.bias'].cpu().numpy()
                Fea_CW1 = C.state_dict()['classifier.0.weight'].cpu().numpy()
                Fea_Cb1 = C.state_dict()['classifier.0.bias'].cpu().numpy()
                scipy.io.savemat(os.path.join(save_dir, 'Feature_W1.mat'), {'array': Fea_W1})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_W2.mat'), {'array': Fea_W2})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_b1.mat'), {'array': Fea_b1})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_b2.mat'), {'array': Fea_b2})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_CW1.mat'), {'array': Fea_CW1})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_Cb1.mat'), {'array': Fea_Cb1})
                scipy.io.savemat(os.path.join(save_dir, 'Pred_all.mat'), {'array': pred_all.cpu().data.numpy()})
                scipy.io.savemat(os.path.join(save_dir, 'Feature_all.mat'), {'array': fea_all.cpu().data.numpy()})
                scipy.io.savemat(os.path.join(save_dir, 'All_reconstruct.mat'),
                                 {'array': All_reconstruct.cpu().data.numpy()})


