## Style-VT: Style Conditioned Chord Generation by Variational Transformer with Chord Substitution Theory

Recent deep learning approaches for melody harmonization have demonstrated strong performance in generating appropriate chords for melodies. However, many of these approaches have not attempted to control chord progressions to adjust the mood of the music, which can significantly influence the music style even for an identical melody. We propose the Chord Generation Variational Transformer model (CGVT), which can capture the structure of chord progressions commonly used in specific genres, thus enabling the generation of chord progressions suited to the style of the desired genre.
Additionally, we address the issue arising from previous studies that recognize only a single chord as the correct answer by applying chord substitution theory. By adjusting the loss function to account for chords that have similar qualities within the context of chord progressions, we facilitate faster convergence during model training and enable the generation of more flexible chord progressions.

## PARSE DATA

1) CMD: 
- download raw data at: https://github.com/shiehn/chord-melody-dataset
- save raw data as: ./CMD/dataset/abc...
- run command: python -m process_data 

2) MuseScore

outputs:
1) saves npy files for the parsed features (saved directory ex: ./CMD/output/~) 
2) saves train/val/test batches (saved directory ex: ./CMD/exp/train/batch/~)
3) saves h5py dataset for the train/val/test batches (saved filename ex: ./CMD_train.h5)


## TRAIN MODEL

1) CGVT 
python -m trainer 

outputs:
1) model parameters/losses checkpoints (saved filename ex: ./trained/STHarm_CMD)


## TEST MODEL 
python -m test [dataset] [song_ind] [start_point] [model_name] [device_num] [alpha]

* [dataset] -> CMD or HLSD 
* [song_ind] -> index of test files 
* [start_point] -> half-bar index ex) if you want to start at the 1st measure: start_point=0 / at the second half of the 1st measure: start_point=1
* [model_name] -> STHarm / VTHarm / rVTHarm 
* [device_num] -> number of CUDA_DEVICE
* [alpha] -> alpha value for rVTHarm / if not fed (different model), ignored

outputs:
1) prints out quantitative metric results -> CHE, CC, CTR, PCS, CTD, MTD / LD, TPSD, DICD 
2) saves generated(sampled) MIDI / GT MIDI (saved filename ex: ./Sampled__*.mid, ./GT__*.mid)
