from scipy import signal
import numpy as np
import mne
import scipy.io as scio
import argparse
import os


def preprocessing(data, fs, Fstop1, Fstop2):
    filtedData = data
    filtedData = signal.detrend(filtedData, axis=1, type='linear')
    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, filtedData, axis=1)
    return filtedData

def preprocessing2(data, fs, Fstop1, Fstop2):
    filtedData = data
    filtedData = signal.detrend(filtedData, axis=1, type='linear', )
    b, a = signal.butter(5, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, filtedData, axis=1)
    return filtedData

def get_source_eeg_BNCI2014001(person_id, current_working_dir, train, resample_fs=250, Fstop1=8, Fstop2=26):
    if train:
        path = current_working_dir + 'A' + f'{person_id:0>2}' + "T.gdf"
    else:
        path = current_working_dir + 'A' + f'{person_id:0>2}' + "E.gdf"
        truth_path = current_working_dir + "A0" + str(person_id) + "E.mat"
    rawDataGDF = mne.io.read_raw_gdf(path, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right']) 
    fs = int(rawDataGDF.info['sfreq'])
    event_position = rawDataGDF.annotations.onset  
    event_type = rawDataGDF.annotations.description  
    temp = rawDataGDF.to_data_frame().drop(['time'], axis=1)
    chan_time_all = temp.T.to_numpy()
    pre_data = preprocessing(chan_time_all, fs, Fstop1, Fstop2)
    if resample_fs != fs:
        pre_data = signal.resample(pre_data, int(pre_data.shape[1] / (fs / resample_fs)), axis=1)
        fs = resample_fs
    if train:
        mi_left = []
        mi_right = []
        mi_foot = []
        mi_tongue = []
        for xuhao, type_mi in enumerate(event_type):
            event_start_position = int(event_position[xuhao] * fs)
            if type_mi == '769':
                mi_left.append(pre_data[:, event_start_position:event_start_position + fs * 4])
            elif type_mi == '770':
                mi_right.append(pre_data[:, event_start_position:event_start_position + fs * 4])
            elif type_mi == '771':
                mi_foot.append(pre_data[:, event_start_position:event_start_position + fs * 4])
            elif type_mi == '772':
                mi_tongue.append(pre_data[:, event_start_position:event_start_position + fs * 4])
        ratio = 5/6
        mi_left_len = int(len(mi_left)*ratio)
        mi_right_len = int(len(mi_right)*ratio)
        mi_foot_len = int(len(mi_foot)*ratio)
        mi_tongue_len = int(len(mi_tongue)*ratio)
        train_data = np.vstack((mi_left[:mi_left_len], mi_right[:mi_right_len], mi_foot[:mi_foot_len], mi_tongue[:mi_tongue_len]))
        train_label = np.vstack((np.zeros(mi_left_len, dtype=int), np.ones(mi_right_len, dtype=int),
                                 2 * np.ones(mi_foot_len, dtype=int), 3 * np.ones(mi_tongue_len, dtype=int))).reshape(-1)
        validate_data = np.vstack((mi_left[mi_left_len:], mi_right[mi_right_len:], mi_foot[mi_foot_len:], mi_tongue[mi_tongue_len:]))
        validate_label = np.vstack((np.zeros(len(mi_left)-mi_left_len, dtype=int), np.ones(len(mi_right)-mi_right_len, dtype=int),
                                    2 * np.ones(len(mi_foot)-mi_foot_len, dtype=int), 3 * np.ones(len(mi_tongue)-mi_tongue_len, dtype=int))).reshape(-1)
        return train_data, train_label, validate_data, validate_label
    else:
        mi_unknown = []
        for xuhao, type_mi in enumerate(event_type):
            if type_mi == '783':  
                event_start_position = int(event_position[xuhao] * fs)
                mi_unknown.append(pre_data[:, event_start_position:event_start_position + fs * 4])
        test_data = np.array(mi_unknown)
        test_label = np.array(scio.loadmat(truth_path)['classlabel']) - 1
        return test_data, test_label

def get_source_eeg_BNCI2014002(person_id, current_working_dir, train, resample_fs=128, Fstop1=8, Fstop2=26):
    if train:
        position_list = [0, 1, 2, 3, 4]
        data_path = working_dir + 'S' + f'{person_id:0>2}' + "T.mat"
        data = scio.loadmat(data_path)['data']
        mi_left = []
        mi_right = []
        for position in position_list:
            fs = data[0, position]['fs'][0, 0][0, 0]
            trial = data[0, position]['trial'][0, 0]
            label = data[0, position]['y'][0, 0] - 1
            eeg = data[0, position]['X'][0, 0].T
            pre_data = preprocessing2(eeg, fs, Fstop1, Fstop2)
            if resample_fs != fs:
                pre_data = signal.resample(pre_data, int(pre_data.shape[1] / (fs / resample_fs)), axis=1)
                trial = trial / fs * resample_fs
                fs = resample_fs
            for index, trial_start in enumerate(trial[0, :]):
                trial_start = int(trial_start)
                eeg_temp = pre_data[:, trial_start + fs * 4:trial_start + fs * 8]
                label_temp = label[0, index]
                if label_temp == 0:
                    mi_left.append(eeg_temp)
                elif label_temp == 1:
                    mi_right.append(eeg_temp)
        ratio = 5/6 
        mi_left_len = int(len(mi_left)*ratio)
        mi_right_len = int(len(mi_right)*ratio)
        train_data = np.vstack((mi_left[:mi_left_len], mi_right[:mi_right_len]))
        train_label = np.vstack((np.zeros(mi_left_len, dtype=int), np.ones(mi_right_len, dtype=int))).reshape(-1)
        validate_data = np.vstack((mi_left[mi_left_len:], mi_right[mi_right_len:]))
        validate_label = np.vstack((np.zeros(len(mi_left) - mi_left_len, dtype=int), np.ones(len(mi_right) - mi_right_len, dtype=int))).reshape(-1)
        return train_data, train_label, validate_data, validate_label
    else:
        position_list = [0, 1, 2]
        data_path = current_working_dir + 'S' + f'{person_id:0>2}' + "E.mat"
        data = scio.loadmat(data_path)['data']
        mi_left = []
        mi_right = []
        for position in position_list:
            fs = data[0, position]['fs'][0, 0][0, 0]
            trial = data[0, position]['trial'][0, 0]
            label = data[0, position]['y'][0, 0] - 1
            eeg = data[0, position]['X'][0, 0].T
            pre_data = preprocessing2(eeg, fs, Fstop1, Fstop2)
            if resample_fs != fs:
                pre_data = signal.resample(pre_data, int(pre_data.shape[1] / (fs / resample_fs)), axis=1)
                trial = trial / fs * resample_fs
                fs = resample_fs
            for index, trial_start in enumerate(trial[0, :]):
                trial_start = int(trial_start)
                eeg_temp = pre_data[:, trial_start + fs * 4:trial_start + fs * 8]
                label_temp = label[0, index]
                if label_temp == 0:
                    mi_left.append(eeg_temp)
                elif label_temp == 1:
                    mi_right.append(eeg_temp)
        mi_left_len = len(mi_left)
        mi_right_len = len(mi_right)
        test_data = np.vstack((mi_left[:mi_left_len], mi_right[:mi_right_len]))
        test_label = np.vstack((np.zeros(mi_left_len, dtype=int), np.ones(mi_right_len, dtype=int))).reshape(-1)
        return test_data, test_label

def get_source_eeg_from_Weibo2014(person_id, current_working_dir, train, resample_fs=128, Fstop1=8, Fstop2=26):
    data_path = current_working_dir + f'subject_{person_id}.mat'
    eeg, label = scio.loadmat(data_path)['data'], scio.loadmat(data_path)['label']
    fs = 200
    channels, samples, trials = eeg.shape
    pre_data = preprocessing(eeg, fs, Fstop1, Fstop2)
    if resample_fs != fs:
        pre_data = signal.resample(pre_data, int(pre_data.shape[1] / (fs / resample_fs)), axis=1)
        fs = resample_fs
    left_hand = []
    right_hand = []
    hands = []
    feet = []
    left_hand_right_foot = []
    right_hand_left_foot = []
    rest = []
    channel_list = list(range(62))
    channel_list.remove(61)
    channel_list.remove(57)
    for idx in range(trials):
        eeg_temp = pre_data[channel_list, fs * 3:fs * 7, idx]
        label_temp = label[idx, 0]
        if label_temp == 1:
            left_hand.append(eeg_temp)
        elif label_temp == 2:
            right_hand.append(eeg_temp)
        elif label_temp == 3:
            hands.append(eeg_temp)
        elif label_temp == 4:
            feet.append(eeg_temp)
        elif label_temp == 5:
            left_hand_right_foot.append(eeg_temp)
        elif label_temp == 6:
            right_hand_left_foot.append(eeg_temp)
        elif label_temp == 7:
            rest.append(eeg_temp)
    if train:
        left_hand = left_hand[:len(left_hand) // 2]
        right_hand = right_hand[:len(right_hand) // 2]
        hands = hands[:len(hands) // 2]
        feet = feet[:len(feet) // 2]
        left_hand_right_foot = left_hand_right_foot[:len(left_hand_right_foot) // 2]
        right_hand_left_foot = right_hand_left_foot[:len(right_hand_left_foot) // 2]
        ratio = 5 / 6  
        left_hand_len = int(len(left_hand) * ratio)
        right_hand_len = int(len(right_hand) * ratio)
        hands_len = int(len(hands) * ratio)
        feet_len = int(len(feet) * ratio)
        left_hand_right_foot_len = int(len(left_hand_right_foot) * ratio)
        right_hand_left_foot_len = int(len(right_hand_left_foot) * ratio)
        train_data = np.vstack(
            (left_hand[:left_hand_len], right_hand[:right_hand_len], hands[:hands_len], feet[:feet_len],
             left_hand_right_foot[:left_hand_right_foot_len], right_hand_left_foot[:right_hand_left_foot_len]))
        train_label = np.vstack((np.zeros(left_hand_len, dtype=int), np.ones(right_hand_len, dtype=int),
                                 2 * np.ones(hands_len, dtype=int), 3 * np.ones(feet_len, dtype=int),
                                 4 * np.ones(left_hand_right_foot_len, dtype=int),
                                 5 * np.ones(right_hand_left_foot_len, dtype=int))).reshape(-1)
        validate_data = np.vstack(
            (left_hand[left_hand_len:], right_hand[right_hand_len:], hands[hands_len:], feet[feet_len:],
             left_hand_right_foot[left_hand_right_foot_len:], right_hand_left_foot[right_hand_left_foot_len:]))
        validate_label = np.vstack(
            (np.zeros(len(left_hand) - left_hand_len, dtype=int), np.ones(len(right_hand) - right_hand_len, dtype=int),
             2 * np.ones(len(hands) - hands_len, dtype=int), 3 * np.ones(len(feet) - feet_len, dtype=int),
             4 * np.ones(len(left_hand_right_foot) - left_hand_right_foot_len, dtype=int),
             5 * np.ones(len(right_hand_left_foot) - right_hand_left_foot_len, dtype=int))).reshape(-1)
        return train_data, train_label, validate_data, validate_label
    else:
        left_hand = left_hand[len(left_hand) // 2:]
        right_hand = right_hand[len(right_hand) // 2:]
        hands = hands[len(hands) // 2:]
        feet = feet[len(feet) // 2:]
        left_hand_right_foot = left_hand_right_foot[len(left_hand_right_foot) // 2:]
        right_hand_left_foot = right_hand_left_foot[len(right_hand_left_foot) // 2:]
        test_data = np.vstack((left_hand, right_hand, hands, feet, left_hand_right_foot, right_hand_left_foot))
        test_label = np.vstack(
            (np.zeros(len(left_hand), dtype=int), np.ones(len(right_hand), dtype=int),
             2 * np.ones(len(hands), dtype=int), 3 * np.ones(len(feet), dtype=int),
             4 * np.ones(len(left_hand_right_foot), dtype=int),
             5 * np.ones(len(right_hand_left_foot), dtype=int))).reshape(-1)
        return test_data, test_label

def Zsolve(R):
    v, Q = np.linalg.eig(R)
    ss1 = np.diag(v ** (-0.5))
    ss1[np.isnan(ss1)] = 0
    re = np.dot(Q, np.dot(ss1, np.linalg.inv(Q)))
    return np.real(re)

def EA(X):
    num_trial, num_channel, num_sampls, num_fb = np.shape(X)
    datas = []
    for fb in range(num_fb):
        data = X[:, :, :, fb]
        R = np.zeros((num_channel, num_channel))
        for i in range(num_trial):
            XTemp = np.squeeze(data[i, :, :])
            R = R + np.dot(XTemp, XTemp.T)
        R = R / num_trial
        R = Zsolve(R)
        for i in range(num_trial):
            XTemp = np.squeeze(data[i, :, :])
            XTemp = np.dot(R, XTemp)
            data[i, :, :] = XTemp
        datas.append(data)
    datas = np.stack(datas, axis=-1)
    return datas

def save_data_label(work_path, person_id, train_data, train_label, validate_data, validate_label, test_data, test_label):
    if not os.path.isdir(work_path + 'train/'):
        os.mkdir(work_path + 'train')
    if not os.path.isdir(work_path + 'validate/'):
        os.mkdir(work_path + 'validate')
    if not os.path.isdir(work_path + 'test/'):
        os.mkdir(work_path + 'test')
    if not os.path.isdir(work_path + 'train/EA'):
        os.mkdir(work_path + 'train/EA')
    if not os.path.isdir(work_path + 'validate/EA'):
        os.mkdir(work_path + 'validate/EA')
    if not os.path.isdir(work_path + 'test/EA'):
        os.mkdir(work_path + 'test/EA')
    np.save(work_path + 'train/' + f'data_id{person_id}.npy', train_data)
    np.save(work_path + 'train/' + f'label_id{person_id}.npy', train_label)
    np.save(work_path + 'validate/' + f'data_id{person_id}.npy', validate_data)
    np.save(work_path + 'validate/' + f'label_id{person_id}.npy', validate_label)
    np.save(work_path + 'test/' + f'data_id{person_id}.npy', test_data)
    np.save(work_path + 'test/' + f'label_id{person_id}.npy', test_label)
    train_data = np.load(work_path + 'train/' + f'data_id{person_id}.npy', allow_pickle=True)
    validate_data = np.load(work_path + 'validate/' + f'data_id{person_id}.npy', allow_pickle=True)
    train_validate_eeg = np.concatenate((train_data, validate_data), axis=0)
    train_validate_eeg_EA = EA(train_validate_eeg)
    train_data_EA = train_validate_eeg_EA[:train_data.shape[0]]
    train_label = np.load(work_path + 'train/' + f'label_id{person_id}.npy', allow_pickle=True)
    np.save(work_path + 'train/' + 'EA/' + f'data_id{person_id}.npy', train_data_EA)
    np.save(work_path + 'train/' + 'EA/' + f'label_id{person_id}.npy', train_label)
    validate_eeg_EA = train_validate_eeg_EA[train_data.shape[0]:]
    validate_label = np.load(work_path + 'validate/' + f'label_id{person_id}.npy', allow_pickle=True)
    np.save(work_path + 'validate/' + 'EA/' + f'data_id{person_id}.npy', validate_eeg_EA)
    np.save(work_path + 'validate/' + 'EA/' + f'label_id{person_id}.npy', validate_label)
    test_data = np.load(work_path + 'test/' + f'data_id{person_id}.npy', allow_pickle=True)  # [N, C, T]
    test_eeg_EA = EA(test_data)
    test_label = np.load(work_path + 'test/' + f'label_id{person_id}.npy', allow_pickle=True)
    np.save(work_path + 'test/' + 'EA/' + f'data_id{person_id}.npy', test_eeg_EA)
    np.save(work_path + 'test/' + 'EA/' + f'label_id{person_id}.npy', test_label)


#########   MI EEG 预处理   #########
parser = argparse.ArgumentParser(description='MI EEG preprocess')
parser.add_argument('--dataset', type=int, default=0, help='Choose Dataset')
parser.add_argument('--resample_fs', type=int, default=250, help='Choose fs')
prms = vars(parser.parse_args())

if __name__ == '__main__':
    resample_fs = prms['resample_fs']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = current_dir + '/data/' +  ['BNCI2014001/', 'BNCI2014002/', 'Weibo2014/'][prms['dataset']]
    downsample_preprocess_dir = working_dir + f'{resample_fs}' + 'Hz_preprocess_eeg/' + 'filterbank/'
    if not os.path.isdir(working_dir + f'{resample_fs}' + 'Hz_preprocess_eeg/' + 'filterbank/'):
        os.mkdir(working_dir + f'{resample_fs}' + 'Hz_preprocess_eeg/' + 'filterbank/')
    for person_id in [range(1, 10), range(1, 15), range(1, 11)][prms['dataset']]:
        train_datas, validate_datas, test_datas = [], [], []
        for F in [[4,8], [8,12], [12,16], [16,20], [20,24], [24,28], [28,32], [32,36], [36,40]]:
            train_data, train_label, validate_data, validate_label = [get_source_eeg_BNCI2014001, get_source_eeg_BNCI2014002, get_source_eeg_from_Weibo2014][prms['dataset']](person_id=person_id,
                                                                                        current_working_dir=working_dir,
                                                                                        train=True, resample_fs=resample_fs,
                                                                                        Fstop1=F[0], Fstop2=F[1])
            test_data, test_label = [get_source_eeg_BNCI2014001, get_source_eeg_BNCI2014002, get_source_eeg_from_Weibo2014][prms['dataset']](person_id=person_id, current_working_dir=working_dir,
                                                            train=False, resample_fs=resample_fs, Fstop1=F[0], Fstop2=F[1])
            train_datas.append(train_data)
            validate_datas.append(validate_data)
            test_datas.append(test_data)
        train_datas = np.stack(train_datas, axis=-1)
        validate_datas = np.stack(validate_datas, axis=-1)
        test_datas = np.stack(test_datas, axis=-1)
        save_data_label(work_path=downsample_preprocess_dir, person_id=person_id, train_data=train_datas,
                        train_label=train_label, validate_data=validate_datas, validate_label=validate_label,
                        test_data=test_datas, test_label=test_label)
