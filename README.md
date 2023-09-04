# SARID (Surveillance-Audio-Rainfall-Intensity-Dataset)

# UPdate on 04/09/2023. SARID V.1. 
It can be downloaded from (Google Drive will be available in one month ):
 - [BaiduYun Drive(code: ugpp)](https://pan.baidu.com/s/1WbC-rP8gs54OuwnHzb71bg)

#### train\test split
The split file is available under 'split/' folder.

### metric

- For each audio slice, the baseline outputs only one rainfall intensity value. The MAE, MSE,	RMSE, and	R2 are taken as the metric.

##### baseline results

|   Network   | Acoustic Feature |   MAE  |  MSE |   RMSE  | R2 |
|---|---|---|---|---|---|
|     CNN     |       MFCC       | 0.646 | 1.008 | 1.004 | 0.694 |
|     LSTM    |       MFCC       | 0.713 | 1.197 | 1.094 | 0.637 |
| Transformer |       MFCC       | 0.563 | 0.775 | 0.88  | 0.765 |
|     CNN     |        Mel       | 0.85  | 1.666 | 1.291 | 0.494 |
|     LSTM    |        Mel       | 0.796 | 1.484 | 1.218 | 0.55  |
| Transformer |        Mel       | 0.856 | 1.495 | 1.223 | 0.546 |
|     CNN     |       STFT       | 0.856 | 1.495 | 1.223 | 0.546 |
|     LSTM    |       STFT       | 0.816 | 1.353 | 1.163 | 0.589 |
| Transformer |       STFT       | 0.711 | 1.199 | 1.095 | 0.636 |


This repository is designed to provide an open-source dataset for surveillance audio-based rainfall estimation, described in _《Towards Rainfall Intensity Estimation using Surveillance Audio: A Dataset and Baseline》_. This dataset is open-source under MIT license. More details about this dataset are avialable at our paper _《Towards Rainfall Intensity Estimation using Surveillance Audio: A Dataset and Baseline》_. If you are benefited from this paper, please cite our paper as follows:

```
@inproceedings{xu2018towards,
  title={Towards Rainfall Intensity Estimation using Surveillance Audio: A Dataset and Baseline},
  author={Meizhen WANG, Mingzheng CHEN, Ziran WANG, Yuxuan GUO, Xuejun LIU*},
  booktitle={**},
  pages={**},
  year={2023}
}
```



## Demo

Demo code and several images are provided, after you obtain "model_epoch_best_R.ckpt" by downloading or training, the "dataprocessing.py" is used to process the audio data (including the acoustic feature extraction, tran/test splitting, etc.). The "baseline.py" includes the model structure and training process.

## Training instructions

Input parameters are well commented in python codes(python2/3 are both ok, the version of pytorch should be >= 0.3). You can increase the batchSize as long as enough GPU memory is available.

#### Environment (not so important as long as you can run the code): 

## Dataset Annotations

Annotations are embedded in file name.

A sample image name is "2022-09-15 17-56-00_0.19_22.335_92.35_2.491_0.892_hiv00013_60_road(concrete).mp3.". Each name can be splited into several fields. Those fields are explained as follows.

- **2022-09-15 17-56-00**: Time annotation indicating the recording time as September 15th, 2022, at 17:56.
- **0.19**: Represents the rainfall intensity in millimeters per hour during the specific time interval.
- **22.335**: Indicates the average temperature in Celsius for the corresponding time period.
- **92.35**: Represents the average humidity observed during the recorded time interval, expressed as a percentage.
- **2.491**: Indicates the average atmospheric pressure in hPa during the given time segment.
- **0.892**: Indicates the average wind speed recorded in meters per second within the specific time duration.
- **hiv00013**: Indicates the original video file associated with the current audio segment.
- **60**: Represents the total duration of the current audio file, which is 60 seconds.
- **road(concrete)**: Specifies the underlying surface as a concrete road for the recording.

## Acknowledgement

If you have any problems about SARID, please contact chenmingzheng64@gmail.com.

Please cite the paper _《Towards Rainfall Intensity Estimation using Surveillance Audio: A Dataset and Baseline》_, if you benefit from this dataset.







