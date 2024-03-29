# Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository contains various artifacts, such as source code, experimental results, and other materials, that supplement our work on **Substructure-aware Log Anomaly Detection**. This work proposes a novel framework for log anomaly detection, Substructure-aware Log Anomaly Detection based on Graph Neural Networks (SLAD). It first introduces a Monte Carlo tree search strategy tailored specifically for log anomaly detection to discover representative substructures. Then, SLAD incorporates a substructure distillation way to enhance the efficiency of anomaly inference based on the representative substructures. The framework of SLAD is listed below.
![SLAD framework diagram](https://github.com/zzxhelloworld/SLAD/blob/main/Artifact/slad-framework.png)
# Preliminaries: Getting datasets ready for experiments
> 1. Download 2 datasets: Forum & Halo
>> The datasets for our experiments are placed [here](https://drive.google.com/drive/folders/11blVvVFOB7LHcf8IkZlzEUD08J7eGBys?usp=drive_link). The datasets are in a .rar format. Before conducting experiments,  please unzip the datasets and place them in the same directory with the source code.
> 2. Download other necessary files for experiments
>> Please visit the directory <kbd>Artifact/necessary/</kbd> and download all of them. Please make sure to place them in the same directory with the source code.
> 3. Software requirements
>>python 3.10+; pytorch 2.1.2+; torch-geometric 2.4.0+
# How to run code from the command line
1. Clone the repository:
   ```bash
   git clone https://github.com/zzxhelloworld/SLAD.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <your_project_directory>
   ```
3. Run code from the command line for experiments:
   ```bash
   python3 --bs <batch size> --n_epoch <epoch number> --representation_num <number of representative substructures> --dataset <Forum/Halo> slad_framework.py
   ```

   

