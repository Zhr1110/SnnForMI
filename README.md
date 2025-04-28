# SnnForMI
This repository is the implementation of "A Lightweight Spiking Neural Network for EEG-Based Motor Imagery Classification".
1. First, run preprocess.py to process the three datasets respectively.
2. Next, run preprocess_for_filterbank.py to obtain the multi-band versions of the three datasets (for FBCNet).
3. Lastly, run main.py to conduct within-subject and cross-subject experiments for motor imagery.
4. Note that SNNs.py contains our proposed model 'CUPY_SNN_PLIF', the remaining files are support files for the baseline or SNN, and the environment required is included in the requirements.txt.
5. This implementation has been based on SpikingJelly (https://github.com/fangwei123456/spikingjelly).