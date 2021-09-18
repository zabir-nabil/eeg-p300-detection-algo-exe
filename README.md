# eeg-p300-detection-algo-exe
An executable (.exe) from a python script for p300 segment extraction using different channels

### data
The program requires a csv file with different eeg channels.

input_eeg_data.csv
```
TimeStamp,Delta_TP9,Delta_AF7,Delta_AF8,Delta_TP10,Theta_TP9,Theta_AF7,Theta_AF8,Theta_TP10,Alpha_TP9,Alpha_AF7,Alpha_AF8,Alpha_TP10,Beta_TP9,Beta_AF7,Beta_AF8,Beta_TP10,Gamma_TP9,Gamma_AF7,Gamma_AF8,Gamma_TP10,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10,AUX_RIGHT,Accelerometer_X,Accelerometer_Y,Accelerometer_Z,Gyro_X,Gyro_Y,Gyro_Z,HeadBandOn,HSI_TP9,HSI_AF7,HSI_AF8,HSI_TP10,Battery
15:06.0,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,846.1539,791.75824,800.2198,840.5128,1152.381,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60
15:06.0,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,732.1245,803.0403,769.19415,772.8205,1121.7583,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60
15:06.0,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,650.32965,803.44324,753.8828,702.71063,862.27106,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60
15:06.0,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,749.45056,794.57874,790.9524,745.0183,750.65936,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60
15:06.2,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,871.9414,796.1905,793.7729,834.87177,896.9231,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60
15:06.0,0.27492267,0.46229774,0.621691,0.44646338,0.6346841,-0.2122276,-0.001418099,0.3284156,0.85608107,0.456067,0.27066702,0.45449367,0.8406303,0.5690032,0.47438997,0.488851,0.36848024,0.27840284,0.34414506,0.08125826,796.1905,808.2784,785.7143,804.65204,884.83514,-0.404663086,0.106628418,0.933410645,2.093505859,-4.725341797,-1.525268555,1,1,1,1,1,60

```
