# Active Learning Methods for Highthroughput Data


We implemented three models here: (1) Uncertainty sampling, (2) Query-by-Committee and (2) Deep Reinforcement based Active Learning model. <br>
The codes for each of them were written separately, and therefore are implemented separately. In the end, the results for each of them are consolidated and the plotting.ipynb file can be run.

Both uncertainty sampling and QBC codes can be implemented directly on the terminal without any changes to the code. It takes around an hour for each of them to complete 10 simulations.

In order to run the rein_model.py file, it is suggested to implement this on AWS or Cluster. Paramters need to be initialized in the file. In order to do so, open the file and scroll down to the "Make Changes Here" section.<br>
Specify the location of the data.csv file and the location where the output needs to be stored. Set the required hyperparameters.<br> 
This should be enough to dry run the model. It takes around 30 minutes for 10 simulations to be completed using GPU.
