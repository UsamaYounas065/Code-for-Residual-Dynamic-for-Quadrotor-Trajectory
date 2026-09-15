# NeuroBEM: Hybrid Aerodynamic Quadrotor Modeling
## Dataset
### Introduction
We release our extensive dataset which contains recordings from 1h:15min of agile quadrotor flights. The dataset was recorded with a custom-built drone in a large hangar equipped with a Vicon motion-capture system. At a rate of 400 Hz, we provide accurate measurements of
- position (Vicon, millimeter precision)
- velocity (filtered from Vicon)
- acceleration (filtered using both onboard and Vicon measurments)
- pose measurements (Vicon)
- body rates (filtered using both onboard and Vicon measurements)
- angular acceleration (filtered using both onboard and Vicon measurements)
- battery voltage of the quadrotor
- individual motor speeds 
- derivative of individual motor speeds

The inertial frame used in this work has a z-axis pointing upwards. Similarly, the body frame of the drone is a front-left-up frame, i.e. x points forwards, y to the left and the z-axis points in the direction of the propeller thrust.

### Folder Structure
An overview over all flights is given in `Flights.txt`.
The data is provided in seperate folders with the following content:
- `processed_data/` contains the main dataset
- `raw_data/` contains the raw measurements of all our flights
- `pdf/` contains plots of all flights
- `code/` contains all the code along with an example dataset

### Processed Data
This folder contains the main dataset. For each flight listed in `Flights.txt` one or multiple csv files exist. This happens because the flights are subdivided so that only regions where the drone is actually flying are contained in dataset. Furthermore, sections where the Vicon systems had dropouts are removed from the data to ensure consistency and high data quality. To see which segment of a flight is used (and what its index is), check the corresponding plots in the `pdf/` subfolder.

Each csv file in the folder has the following column order which is also specified in each files' header.
| Column | Quantity                              | Header Abbreviation |
|--------|---------------------------------------|---------------------|
| 1      | time [s]                              | t                   |
| 2      | angular acceleration body x [rad/s^2] | ang acc x           |
| 3      | angular acceleration body y [rad/s^2] | ang acc y           |
| 4      | angular acceleration body z [rad/s^2] | ang acc z           |
| 5      | angular velocity body x [rad/s]       | ang vel x           |
| 6      | angular velocity body y [rad/s]       | ang vel y           |
| 7      | angular velocity body z [rad/s]       | ang vel z           |
| 8      | quaternion qx                         | qx                  |
| 9      | quaternion qy                         | qy                  |
| 10     | quaternion qz                         | qz                  |
| 11     | quaternion qw                         | qw                  |
| 12     | acceleration body x [m/s^2]           | acc x               |
| 13     | acceleration body y [m/s^2]           | acc y               |
| 14     | acceleration body z [m/s^2]           | acc z               |
| 15     | velocity body x [m/s]                 | vel x               |
| 16     | velocity body y [m/s]                 | vel y               |
| 17     | velocity body z [m/s]                 | vel z               |
| 18     | position x [m]                        | pos x               |
| 19     | position y [m]                        | pos y               |
| 20     | position z [m]                        | pos z               |
| 21     | motor speed back right [rad/s]        | mot 1               |
| 22     | motor speed front right [rad/s]       | mot 2               |
| 23     | motor speed back left  [rad/s]        | mot 3               |
| 24     | motor speed front left [rad/s]        | mot 4               |
| 25     | derivative motor speed 1 [rad/s^2]    | dmot 1              |
| 26     | derivative motor speed 2 [rad/s^2]    | dmot 2              |
| 27     | derivative motor speed 3 [rad/s^2]    | dmot 3              |
| 28     | derivative motor speed 4 [rad/s^2]    | dmot 4              |
| 29     | battery voltage [V]                   | vbat                |

### Raw Data
The `raw_data/` folder contains two types of files: 
- rosbags from the Vicon motion capture system
- bfl-logs from the betaflight inner loop controller

The files are only provided for completeness, but they do not contain additional information that would be needed. To decode the betaflight logs, use the *blackbox-tools* utility that can be found on GitHub: https://github.com/betaflight/blackbox-tools/

### Predictions Data
The `predictions/` folder contains the same files as the processed data folder, however there are 12 columns appended to the data (also the files contain no header).
| Column | Quantity                     |
|--------|------------------------------|
| 30     | predicted force body x [N]   |
| 31     | predicted force body y [N]   |
| 32     | predicted force body z [N]   |
| 33     | predicted torque body x [Nm] |
| 34     | predicted torque body y [Nm] |
| 35     | predicted torque body z [Nm] |
| 36     | residual force body x [N]    |
| 37     | residual force body y [N]    |
| 38     | residual force body z [N]    |
| 39     | residual torque body x [Nm]  |
| 40     | residual torque body y [Nm]  |
| 41     | residual torque body z [Nm]  |

Our drone has a mass of 0.772 kg and the diagonal elements of the inertia matrix are [0.0025, 0.0021, 0.0043]. The residuals are only provided for convenience as they can be computed based on the the measured accelerations (columns 2,3,4 and 12,13,14) and the predicted forces/torques (columns 30-36).

### Code
Please see the corresponding Readme inside the code-folder.

### PDF's
To quickly check how a specific flight looks and how it is divided into segments, one can check the PDF folder. For each flight in the `Flights.txt` file, one can find a corresponding pdf. It contains plots of all the variables. Furthermore, dashed lines show which segments belong to which part of the flight. The PDF's are just a convenience feature.
