# basic libs
import numpy as np
import json
import os
import gc
import random
from scipy import signal
import cv2
import albumentations as A

# pytorch
import torch
from torch.utils.data import Dataset

#custom modules
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, patients, aug,downsample):

        self.patients = patients
        self.images_list = []

        for patient in patients:
            images = [i[:-4] for i in os.listdir(f'./data/brain_images/{patient}') if i.find('_mask')==-1]
            for image in images:
                self.images_list.append(f'./data/brain_images/{patient}/{image}')





        self.preprocessing = Preprocessing(aug)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        X, y,X_s,y_s= self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        X_s = torch.tensor(X_s, dtype=torch.float)
        y_s = torch.tensor(y_s, dtype=torch.float)

        return X, y,X_s,y_s

    def load_data(self, id, train=True):

        X = cv2.imread(self.images_list[id] + '.tif', cv2.IMREAD_COLOR)

        y = cv2.imread(self.images_list[id] + '_mask.tif', cv2.IMREAD_COLOR)
        y_ = y.copy()
        X,y = self.preprocessing.run(X=X,y=y)

        #second head
        sampled_patient = np.random.uniform(size=1)[0]
        if sampled_patient >= 0.5:

            #NOT the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id)==-1]
            X_s = cv2.imread(np.random.choice(np.array(images_subset)) + '.tif', cv2.IMREAD_COLOR)
            y_s = [0]
        else:
            # the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id) != -1]
            images_subset.remove(self.images_list[id])
            X_s = cv2.imread(np.random.choice(np.array(images_subset)) + '.tif', cv2.IMREAD_COLOR)
            y_s = [1]


        X_s, y_ = self.preprocessing.run(X=X_s, y=y_)

        return X,y,X_s,y_s


class Dataset_test(Dataset_train):
    def __init__(self, patients):
        super().__init__(patients=patients,aug=False,downsample=False)

    def __getitem__(self, idx):

        X, y,X_s,y_s = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X

class Preprocessing():

    def __init__(self,aug):

        self.aug = aug
        self.augmentations = Augmentations(0.5)

        # 7 Hz Hamming, 101th order, Fs = 480Hz
        self.weights_LPF = torch.Tensor([0.000478357075602025,0.00047919948536654457,0.00048124838592506305,0.0004845047590715597,0.0004889690321086591,0.0004946410771334543,0.0005015202105831596,0.0005096051930410375,0.0005188942293029063,0.0005293849687044141,0.0005410745057091392,0.0005539593807574481,0.00056803558137593,0.0005832985435470763,0.0005997431533387784,0.0006173637487930674,0.000636154122073404,0.000656107521869708,0.0006772166560601794,0.0006994736946288481,0.0007228702728376636,0.0007473974946518098,0.0007730459364168141,0.0007998056507858897,0.0008276661708958308,0.0008566165147896676,0.0008866451900841599,0.0009177401988800859,0.0009498890429131789,0.0009830787289434293,0.0010172957743803737,0.0010525262131418538,0.0010887556017436394,0.001125969025617175,0.001164151105652615,0.0012032860049641884,0.001243357435874846,0.0012843486671169915,0.001326242531246069,0.00136902143226358,0.001412667353446084,0.001457161865376584,0.0015024861341746193,0.0015486209299212868,0.001595546635275316,0.0016432432542762305,0.0016916904213305258,0.0017408674103767258,0.0017907531442250685,0.0018413262040674987,0.0018925648391535474,0.0019444469766276358,0.0019969502315231795,0.002050051916908908,0.002103729054182602,0.0021579583835075237,0.002212716374386604,0.0022679792363694898,0.0023237229298874317,0.00237992317721094,0.0024365554735250876,0.002493595098117261,0.0025510171256721234,0.0026087964376684696,0.00266690773387265,0.0027253255439231394,0.002784024239000809,0.00284297804357942,0.002902161047250795,0.002961547216619105,0.0030211104072586587,0.003080824375729563,0.0031406627916455813,0.003200599249788493,0.0032606072822632315,0.0033206603706880575,0.003380731958414013,0.003440795462767857,0.0035008242873127097,0.0035607918341205953,0.003620671516051065,0.0036804367690300905,0.003740061064323404,0.0037995179207984744,0.0038587809171692798,0.0039178237042181065,0.003976620016988535,0.004035143686943842,0.004093368654085042,0.004151268979022815,0.0042088188549975715,0.004265992619841944,0.004322764767880033,0.004379109961757716,0.004435003044198427,0.0044904190496787705,0.004545333216018456,0.004599720995878991,0.004653558068165689,0.004706820349327546,0.004759484004549587,0.004811525458832392,0.004862921407953453,0.004913648829305197,0.0049636849926044755,0.005013007470468404,0.0050615941488515335,0.005109423237339337,0.005156473279293141,0.0052027231618416055,0.0052481521257140345,0.00529273977491078,0.005336466086206147,0.005379311418479236,0.005421256521868292,0.005462282546744154,0.005502371052498543,0.005541504016142972,0.0055796638407141766,0.005616833363482018,0.005652995863955993,0.005688135071686447,0.005722235173856831,0.0057552808226633,0.005787257142478183,0.0058181497367938585,0.005847944694943726,0.005876628598597054,0.005904188528024606,0.005930612068132019,0.0059558873142580805,0.005980002877735094,0.006002947891208692,0.00602471201371456,0.006045285435509592,0.0060646588826552255,0.00608282362135071,0.006099771462014263,0.006115494763110155,0.006129986434719867,0.006143239941855656,0.006155249307514877,0.006166009115473662,0.006175514512818572,0.006183761212215046,0.0061907454939115055,0.006196464207478235,0.006200914773280104,0.006204095183682544,0.006206004003990063,0.006206640373116999,0.006206004003990063,0.006204095183682544,0.006200914773280104,0.006196464207478235,0.0061907454939115055,0.006183761212215046,0.006175514512818572,0.006166009115473662,0.006155249307514877,0.006143239941855656,0.006129986434719869,0.006115494763110155,0.006099771462014263,0.00608282362135071,0.006064658882655226,0.006045285435509593,0.0060247120137145605,0.006002947891208692,0.005980002877735094,0.0059558873142580805,0.005930612068132021,0.005904188528024607,0.005876628598597055,0.005847944694943726,0.0058181497367938585,0.005787257142478184,0.005755280822663301,0.005722235173856832,0.005688135071686448,0.005652995863955993,0.0056168333634820196,0.005579663840714177,0.005541504016142974,0.0055023710524985435,0.005462282546744156,0.005421256521868293,0.005379311418479237,0.005336466086206148,0.00529273977491078,0.005248152125714035,0.005202723161841606,0.005156473279293142,0.0051094232373393375,0.005061594148851534,0.005013007470468404,0.004963684992604476,0.004913648829305197,0.004862921407953454,0.004811525458832392,0.004759484004549589,0.004706820349327547,0.00465355806816569,0.004599720995878994,0.004545333216018457,0.004490419049678772,0.004435003044198427,0.004379109961757717,0.004322764767880033,0.004265992619841946,0.0042088188549975715,0.004151268979022817,0.004093368654085042,0.004035143686943843,0.0039766200169885375,0.003917823704218108,0.003858780917169282,0.003799517920798475,0.0037400610643234067,0.003680436769030091,0.0036206715160510662,0.0035607918341205957,0.0035008242873127114,0.0034407954627678574,0.003380731958414015,0.003320660370688058,0.0032606072822632328,0.003200599249788493,0.0031406627916455818,0.0030808243757295653,0.00302111040725866,0.0029615472166191073,0.002902161047250796,0.0028429780435794217,0.0027840242390008096,0.002725325543923141,0.002666907733872651,0.002608796437668472,0.0025510171256721242,0.002493595098117263,0.0024365554735250876,0.002379923177210941,0.0023237229298874347,0.0022679792363694915,0.0022127163743866063,0.002157958383507524,0.0021037290541826036,0.002050051916908908,0.0019969502315231816,0.0019444469766276358,0.0018925648391535498,0.0018413262040674987,0.0017907531442250704,0.0017408674103767258,0.001691690421330527,0.0016432432542762305,0.001595546635275317,0.0015486209299212885,0.00150248613417462,0.001457161865376586,0.0014126673534460852,0.001369021432263582,0.0013262425312460702,0.0012843486671169934,0.001243357435874846,0.0012032860049641901,0.001164151105652615,0.0011259690256171758,0.0010887556017436394,0.0010525262131418549,0.0010172957743803737,0.0009830787289434304,0.0009498890429131799,0.0009177401988800866,0.0008866451900841612,0.0008566165147896681,0.0008276661708958318,0.0007998056507858905,0.0007730459364168151,0.0007473974946518098,0.0007228702728376639,0.0006994736946288481,0.00067721665606018,0.000656107521869708,0.0006361541220734043,0.000617363748793068,0.0005997431533387788,0.0005832985435470766,0.0005680355813759303,0.0005539593807574487,0.0005410745057091392,0.0005293849687044148,0.0005188942293029063,0.0005096051930410375,0.0005015202105831596,0.0004946410771334543,0.0004889690321086591,0.0004845047590715597,0.00048124838592506305,0.00047919948536654457,0.000478357075602025])
        self.weights_LPF = self.weights_LPF.view(1, 1, self.weights_LPF.shape[0]).float()
        self.padding_LPF = int((self.weights_LPF.shape[2] - 1) / 2)
        self.padding_LPF = torch.Tensor(np.zeros((self.padding_LPF))).float()

    def run(self,X,y,label_process=True):

        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY).astype(np.float32)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY).astype(np.float32)



        if self.aug:
            X,y = self.augmentations.apply_augs(X,y)


        X = X/ 255
        X = X.reshape(1, X.shape[0], X.shape[1])
        #X = X**(1.5)#higher contrast

        y = y / 255
        y = y.reshape(1, y.shape[0], y.shape[1])

        if label_process:
            return X,y
        else:
            return X

    def FIR_filt(self, input, weight, padding_vector):
        input = torch.cat((input, padding_vector), 0)
        input = torch.cat((padding_vector, input), 0)
        input = input.view(1, 1, input.shape[0])
        output = torch.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        output = output.view(output.shape[2])
        return output


    @staticmethod
    def apply_amplitude_scaling(X, y):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if y['rpeaks']:
            #for channel_rpeaks in y['rpeaks']:
            if y['rpeaks'][0]:
                #remove baseline
                for i in range(12):
                    X[:,0] -= np.median(X[:,0])
                return X / np.median(X[y['rpeaks'][0], 0] + 0.001)

        for i in range(12):
            X[:, 0] -= np.median(X[:, 0])

        return X / (X[:,0].std() + 0.001)

    def medfilt(self,x, k):
        """Apply a length-k median filter to a 1D array x.
        Boundaries are extended by repeating endpoints.
        """
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i + 1)] = x[j:]
            y[-j:, -(i + 1)] = x[-1]
        return np.median(y, axis=1)

    def apply_augmentation(self, waveform, meta_data, fs_training):

        # Random resample
        # waveform = self._random_resample(waveform=waveform, meta_data=meta_data,
        #                                  fs_training=fs_training, probability=0.25)

        # Random amplitude scale
        waveform = self._random_scale(waveform=waveform, probability=0.5)

        # Apply synthetic noise
        #waveform = self._add_synthetic_noise(waveform=waveform, fs_training=fs_training, probability=0.25)

        return waveform

    def _random_resample(self, waveform, meta_data, fs_training, probability):
        """Randomly resample waveform.
        bradycardia=3, sinus bradycardia=20, sinus tachycardia=22
        """
        if (
            meta_data['hr'] != 'nan'
            and all(meta_data['labels_training_merged'][label] == 0 for label in [3, 20, 22])
            and self._coin_flip(probability=probability)
        ):
            # Get waveform duration
            duration = waveform.shape[0] / fs_training

            # Get new heart rate
            hr_new = int(meta_data['hr'] * np.random.uniform(1, 1.25))
            if hr_new > 300:
                hr_new = 300
            elif hr_new < 40:
                hr_new = 40
            else:
                pass

            # Get new duration
            duration_new = duration * meta_data['hr'] / hr_new

            # Get number of samples
            samples = int(duration_new * fs_training)

            # Resample waveform
            waveform = signal.resample_poly(waveform, samples, waveform.shape[0], axis=0).astype(np.float32)

            return waveform
        else:
            return waveform

    def _random_scale(self, waveform, probability):
        """Apply random scale factor between 0.25 and 3 to the waveform amplitudes."""
        # Get random scale factor
        scale_factor = random.uniform(0.25, 3.0)

        if self._coin_flip(probability):
            return waveform * scale_factor
        return waveform

    def _add_synthetic_noise(self, waveform, fs_training, probability):
        """Add different kinds of synthetic noise to the signal."""
        waveform = waveform.squeeze()
        for idx in range(waveform.shape[1]):
            waveform[:, idx] = self._generate_baseline_wandering_noise(
                waveform=waveform[:, idx], fs=fs_training, probability=probability
            )
            waveform[:, idx] = self._generate_high_frequency_noise(
                waveform=waveform[:, idx], fs=fs_training, probability=probability
            )
            waveform[:, idx] = self._generate_gaussian_noise(
                waveform=waveform[:, idx], probability=probability
            )
            waveform[:, idx] = self._generate_pulse_noise(waveform=waveform[:, idx], probability=probability)
        return waveform

    def _generate_baseline_wandering_noise(self, waveform, fs, probability):
        """Adds baseline wandering to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):

            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Get number of baseline signals
            baseline_signals = random.randint(1, 5)

            # Loop through baseline signals
            for baseline_signal in range(baseline_signals):
                # Add noise
                waveform += random.uniform(0.01, 0.75) * np.sin(
                    2 * np.pi * random.uniform(0.001, 0.5) * time + random.uniform(0, 60)
                )

        return waveform

    def _generate_high_frequency_noise(self, waveform, fs, probability=0.5):
        """Adds high frequency sinusoidal noise to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):
            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Add noise
            waveform += random.uniform(0.001, 0.3) * np.sin(
                2 * np.pi * random.uniform(50, 200) * time + random.uniform(0, 60)
            )

        return waveform

    def _generate_gaussian_noise(self, waveform, probability=0.5):
        """Adds white noise noise to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):
            waveform += np.random.normal(loc=0.0, scale=random.uniform(0.01, 0.25), size=len(waveform))

        return waveform

    def _generate_pulse_noise(self, waveform, probability=0.5):
        """Adds gaussian pulse to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):

            # Get pulse
            pulse = signal.gaussian(
                int(len(waveform) * random.uniform(0.05, 0.010)), std=random.randint(50, 200)
            )
            pulse = np.diff(pulse)

            # Get remainder
            remainder = len(waveform) - len(pulse)
            if remainder >= 0:
                left_pad = int(remainder * random.uniform(0.0, 1.0))
                right_pad = remainder - left_pad
                pulse = np.pad(pulse, (left_pad, right_pad), 'constant', constant_values=0)
                pulse = pulse / pulse.max()

            waveform += pulse * random.uniform(waveform.max() * 1.5, waveform.max() * 2)

        return waveform

    @staticmethod
    def _coin_flip(probability):
        if random.random() < probability:
            return True
        return False


class Augmentations():
    def __init__(self,prob):

        self.augs = A.Compose([
                    A.HorizontalFlip(p=prob),
                    A.Rotate(limit=10, p=prob),
                    A.RandomSizedCrop(min_max_height=(220, 240), height=256, width=256, p=prob)
            ])



    def apply_augs(self,image,mask):

        #apply horizontal flip
        augmented = self.augs(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        return image,mask