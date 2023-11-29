# Automated Drosophila embryo microinjection
An Automated microinjection platform for Drosophila embryos
### Hardware Requirements
1. Computer
2. Arduino Nano
3. Custom built incline microscopes
4. XYZ stage
5. DSLR camera 
6. Air prssure line
7. Electric pressure regulator

### Software Requirements
The following libraries are used in the Autoinjector software (see install instructions for how to install). 
1. [Ananconda 3](https://repo.anaconda.com/archive/)
2. [Microsoft Visual C++ 2019 Redistributable and Microsoft Build Tools 2019](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019)
3. [NVIDIA driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)
4. [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
5. [cUDNN](https://developer.nvidia.com/rdp/cudnn-archive)
6. [DigiCamControl](https://digicamcontrol.com/download)
7. [Arduino](https://www.arduino.cc/en/software)
8. Packages
    - pip 
    - Native python libraties
      - time
      - sys
      - os
      - user
    - tensorflow-gpu==2.10.0
    - numpy==1.21.1
    - opencv-python==4.5.5.62
    - multiprocess
    - tk 
    - pyserial

## Install Instructions
-------------
Install the following software to operate the Autoinjector. It is recommended to install the software in the order it is listed. Make sure to run every file as administrator (right click, "Run as administrator")! Otherwise, the install may fail. 

### 1. Anaconda
1. Download Anaconda3-2021.11-Windows-x86_64.exe [here](https://repo.anaconda.com/archive/). 
2. Launch the installer and follow installation instructions on screen.

### 2. Microsoft Visual
1. Download Microsoft Visual [here](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019). 
2. Launch the installer and follow installation instructions on screen.

### 3. NVIDIA driver
1. Download the NVIDIA driver [here](https://www.nvidia.com/Download/index.aspx?lang=en-us). 
2. Launch the installer and follow installation instructions on screen.

### 4. CUDA 
1. Check which version of CUDA you need based on the tensorflow version required [here](https://www.tensorflow.org/install/source_windows)
2. Download CUDA [here](https://developer.nvidia.com/cuda-toolkit-archive). 
3. Launch the installer and follow installation instructions on screen.

### 5. cUDNN 
1. Check which version of cUDNN you need based on the tensorflow version required [here](https://www.tensorflow.org/install/source_windows)
2. Download cUDNN [here](https://developer.nvidia.com/rdp/cudnn-archive). 
3. Follow these steps to install [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installdriver-windows).

### 6. DigiCamControl
1. Download the DigiCamControl [here](https://digicamcontrol.com/download). 
2. Launch the installer and follow installation instructions on screen.
3. Make sure an SD card is in the DSLR

### 7. Arduino
1. Download the DigiCamControl [here](https://www.arduino.cc/en/software). 
2. Launch the installer and follow installation instructions on screen.
3. Upload AmeyArduinoCode.ino

### 8. Packages
Search Anaconda Prompt in the Windows search Bar. Open it, run as administrator, and run the following commands:
  - Tensorflow
    ```
    pip install tensorflow-gpu==2.10.0
    ```
  - Numpy
    ```
    pip install numpy==1.21.1
    ```
  - OpenCV
    ```
    pip install opencv-python==4.5.5.62
    ```
  - Multiprocess
    ```
    pip install multiprocess
    ```
  - Tkinter
    ```
    pip install tk
    ```
  - Pyserial
    ```
    pip install pyserial
    ```
## Running the Application
---------
 
 To run the program, click the file "Robot_GUI_multipipette.py" in the Autoinjector folder. This will launch the GUI. For additional operating instructions see the user manual [here](https://github.com/bsbrl/Andrew-Microinjection/blob/main/Running%20the%20robot.pdf).

## License
-------------
This work is lisenced under the University of Minnesota lisence.
