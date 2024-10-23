# SARID (Surveillance-Audio-Rainfall-Intensity-Dataset)

# UPdate on 04/09/2023. SARID V.1. 
It can be downloaded from (The corresponding video files can only be downloaded from BaiduYun Drive) :
 - [BaiduYun Drive](https://pan.baidu.com/s/1-QcS7Y0O4AroSfikph5-Kg code: mzyx)
 - [Google Drive](https://drive.google.com/drive/folders/1jH2uO8Xk7RgrcbtkDpTxM3BCo-4tQMY5?usp=drive_link)

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


This repository is designed to provide an open-source dataset for surveillance audio-based rainfall estimation, described in _《Estimating rainfall intensity based on surveillance audio and deep-learning》_. This dataset is open-source under MIT license. More details about this dataset are available in our paper _《Estimating rainfall intensity based on surveillance audio and deep-learning》_. If you are benefited from this paper, please cite our paper as follows:

```
@{
  title={Estimating rainfall intensity based on surveillance audio and deep-learning},
  author={Meizhen WANG, Mingzheng CHEN, Ziran WANG, Yuxuan GUO, Xuejun LIU*},
  booktitle={Environmental Science and Ecotechnology},
  volume={22}
}
```



## Demo
1. Run the "Acoustic feature extraction" section in `data_processing.py` to generate the features (in .npy format) and labels (in .csv format).  
2. Run `baseline.py` to train and test the model (details provided below).

Once you've obtained the "model_epoch_best_R.ckpt" file—either by downloading it or training the model yourself—you can refer to the provided demo code and images.  
The `data_processing.py` script is responsible for processing audio data, which includes tasks like acoustic feature extraction and splitting the data into training and testing sets.  
For feature preparation, you only need to run the "Acoustic feature extraction" section in `data_processing.py`.  

The `baseline.py` script contains the model architecture and handles the training process. Since different features (MFCC, Mel, STFT) and networks (CNN, LSTM, Transformer) are supported, you need to adjust certain settings in the code—specifically, the input dimension parameters. Ensure that the model's input dimensions match the feature dimensions. Details on these settings can be found in the code.


## Training instructions

Input parameters are well commented in python codes(python2/3 are both ok, the version of pytorch should be >= 0.3). You can increase the batchSize as long as enough GPU memory is available.

#### Note that the environment is not so important as long as you can run the code. 

## Dataset Annotations

Annotations are embedded in the file name.

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

If you have any problems with SARID, please contact chenmingzheng64@gmail.com.

Please cite the paper _ Meizhen Wang, Mingzheng Chen, Ziran Wang, Yuxuan Guo, Yong Wu, Wei Zhao, Xuejun Liu,
Estimating rainfall intensity based on surveillance audio and deep-learning,
Environmental Science and Ecotechnology,
Volume 22,
2024,
100450,
ISSN 2666-4984,
https://doi.org/10.1016/j.ese.2024.100450.
(https://www.sciencedirect.com/science/article/pii/S2666498424000644)
Abstract: Rainfall data with high spatial and temporal resolutions are essential for urban hydrological modeling. Ubiquitous surveillance cameras can continuously record rainfall events through video and audio, so they have been recognized as potential rain gauges to supplement professional rainfall observation networks. Since video-based rainfall estimation methods can be affected by variable backgrounds and lighting conditions, audio-based approaches could be a supplement without suffering from these conditions. However, most audio-based approaches focus on rainfall-level classification rather than rainfall intensity estimation. Here, we introduce a dataset named Surveillance Audio Rainfall Intensity Dataset (SARID) and a deep learning model for estimating rainfall intensity. First, we created the dataset through audio of six real-world rainfall events. This dataset's audio recordings are segmented into 12,066 pieces and annotated with rainfall intensity and environmental information, such as underlying surfaces, temperature, humidity, and wind. Then, we developed a deep learning-based baseline using Mel-Frequency Cepstral Coefficients (MFCC) and Transformer architecture to estimate rainfall intensity from surveillance audio. Validated from ground truth data, our baseline achieves a root mean absolute error of 0.88 mm h-1 and a coefficient of correlation of 0.765. Our findings demonstrate the potential of surveillance audio-based models as practical and effective tools for rainfall observation systems, initiating a new chapter in rainfall intensity estimation. It offers a novel data source for high-resolution hydrological sensing and contributes to the broader landscape of urban sensing, emergency response, and resilience.
Keywords: Surveillance audio; Rainfall intensity; Dataset; Regression; Deep learning
_, if you benefit from this dataset.







