import torch
import torch.nn as nn
from tools import functional, surrogate
from tools.datasets import EEGset
import argparse
import numpy as np
import random
import os
from torch.utils.data import ConcatDataset
import math
from tools import models

parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network for MI')
parser.add_argument('--dataset', type=int, default=0, help='Choose Dataset')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--epoch', type=int, default=1500, help='Number of epochs for stage one')
parser.add_argument('--epoch2', type=int, default=600, help='Number of epochs for stage two')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--trial_num', type=int, default=8, help='Number of repeated experiments')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--patience', type=int, default=200, help='Early Stop Tolerance')
parser.add_argument('--loo', type=bool, default=False, help='whether cross subject')
parser.add_argument('--EA', type=bool, default=False, help='whether EA')
parser.add_argument('--model', type=str, default='CUPY_SNN_PLIF', help='Choose the model')
parser.add_argument('--T', type=int, default=250 * 4, help='Time step')
parser.add_argument('--prep', type=str, default='250Hz_preprocess_eeg/', help='Choose preprocess method')
parser.add_argument('--device', type=str, default='1', help='Choose GPU')
parser.add_argument('--beta', type=float, default=2, help='Choose model hyper-parameter')
prms = vars(parser.parse_args())

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=10, path=None):
        self.patience = patience
        self.counter = 0
        self.val_min_acc = 0.
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):
        if val_acc < self.val_min_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_min_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def stage_one_train(net, train_set, validate_set, model_path, prms):
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=prms['batch_size'],
        shuffle=True,
        drop_last=True
    )
    validate_data_loader = torch.utils.data.DataLoader(
        dataset=validate_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=prms['lr'], betas=(0.9, 0.999))
    loss_function = nn.CrossEntropyLoss().cuda()
    EarlyStop = EarlyStopping(patience=prms['patience'], path=model_path)
    train_loss = 0
    # stage1
    for epoch in range(prms['epoch']):
        net.train()
        accuracy = 0
        loss0 = 0
        train_num = 0
        num = 0
        for frame, label in train_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
            train_num += label.numel()
            num += 1
            functional.reset_net(net)
        accuracy /= train_num
        loss0 /= num
        train_loss = loss0
        net.eval()
        accuracy = 0
        loss0 = 0
        val_num = 0
        num = 0
        with torch.no_grad():
            for frame, label in validate_data_loader:
                frame = frame.cuda()
                label = label.reshape(-1).cuda()
                out_fr = net(frame)
                loss = loss_function(out_fr, label)
                loss0 += loss
                accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
                val_num += label.numel()
                num += 1
                functional.reset_net(net)
            accuracy /= val_num
            loss0 /= num
        EarlyStop(accuracy, net)
        if EarlyStop.early_stop:
            print("Early stopping at %d epoch" % (epoch))
            break
    net.load_state_dict(torch.load(model_path))
    return net, train_loss

def stage_two_train(net, train_set, validate_set, model_path, train_loss, prms):
    combined_dataset = ConcatDataset([train_set, validate_set])
    combined_data_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=prms['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    net.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
    loss_function = nn.CrossEntropyLoss().cuda()
    for epoch in range(prms['epoch2']):
        net.train()
        loss0 = 0
        train_num = 0
        num = 0
        for frame, label in combined_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num += label.numel()
            num += 1
            functional.reset_net(net)
        loss0 /= num
        torch.save(net.state_dict(), model_path)
        if loss0 < train_loss:
            print("current loss < stage 1 train loss at %d epoch" % (epoch))
            break
    net.load_state_dict(torch.load(model_path))
    return net

def test(net, test_set, model_path, prms):
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    net.load_state_dict(torch.load(model_path))
    test_acc = 0
    test_num = 0
    net.eval()
    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            test_acc += (out_fr.argmax(dim=1) == label).float().sum().item()
            test_num += label.numel()
            functional.reset_net(net)
        test_acc /= test_num
        test_acc = round(test_acc, 4)
    return test_acc

if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/' + ['BNCI2014001/', 'BNCI2014002/', 'Weibo2014/'][prms['dataset']] + prms['prep']
    if prms['model'] in ['FBCNet', 'FBMSNet']: data_path += 'filterbank/'
    if prms['dataset'] == 2 or prms['model'] == 'FBMSNet': torch.set_num_threads(2)
    torch.cuda.set_device('cuda:{}'.format(prms['device']))
    print(prms)
    patience = prms['patience']
    trial_acc = []
    for trial_num in range(prms['trial_num']):
        seedval = prms['seed'] + trial_num
        seed_torch(seedval)
        id_test_acc = []
        for ids in [range(1, 10), range(1, 15), range(1, 11)][prms['dataset']]:
            model_path = data_path + prms['model'] + f'_id{ids}_seed{seedval}' + ('_loo' if prms['loo'] else '') + ('_EA' if prms['EA'] else '') + '.pth'
            if prms['model'] in ['ShallowConvNet', 'deepconv', 'EEGNet']:
                net = getattr(models, prms['model'])(in_channels=[22, 15, 60][prms['dataset']], time_step=prms['T'], classes_num=[4, 2, 6][prms['dataset']]).cuda()
            elif prms['model'] in ['FBCNet', 'FBMSNet']:
                net = getattr(models, prms['model'])(nChan=[22, 15, 60][prms['dataset']], nTime=prms['T'], nClass=[4, 2, 6][prms['dataset']]).cuda()
            elif prms['model'] in ['CUPY_SNN_PLIF']:
                net = getattr(models, prms['model'])(in_channels=[22, 15, 60][prms['dataset']], out_num=[4, 2, 6][prms['dataset']], w=prms['w'], time_step=prms['T'], beta=prms['beta']).cuda()
            elif prms['model'] in ['Conformer']:
                net = getattr(models, prms['model'])(in_channels=[22, 15, 60][prms['dataset']], n_classes=[4, 2, 6][prms['dataset']]).cuda()
            if trial_num == 0 and ids == 1: print(net)
            print(prms['model'] + f'_trial_num{trial_num}_id{ids}')
            train_set = EEGset(root_path=data_path, pick_id=(ids,), settup='train', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'])
            validate_set = EEGset(root_path=data_path, pick_id=(ids,), settup='validate', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'])
            test_set = EEGset(root_path=data_path, pick_id=(ids,), settup='test', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'])
            net, train_loss = stage_one_train(net, train_set, validate_set, model_path, prms)
            net = stage_two_train(net, train_set, validate_set, model_path, train_loss, prms)
            test_acc = test(net, test_set, model_path, prms)
            print(f'the test accuracy is {test_acc}\n')
            id_test_acc.append(100 * test_acc)
        print('seed{} mean is {}'.format(seedval, np.mean(np.array(id_test_acc))))
        trial_acc.append(id_test_acc)
    trial_acc = np.array(trial_acc)
    print(trial_acc)
    id_mean = np.mean(trial_acc, axis=0).reshape(-1)
    id_var = np.var(trial_acc, axis=0).reshape(-1)
    trial_mean = np.mean(trial_acc, axis=1).reshape(-1)
    result_mean = np.mean(trial_mean)
    result_var = np.var(trial_mean)
    print(f'results is: {result_mean}±{np.sqrt(result_var)}')
