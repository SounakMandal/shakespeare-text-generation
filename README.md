# shakespeare_text_generation
This project generates models which learn from sonnets and plays written by shakespeare.
The learned models (newly trained or pretrained) can be used to generate original text
in the style of Shakespeare. Unfortunately not everything the model generates is meaningful,
but largely the model is capable of genrating meaningful text after sufficient training.


## Details
The model works by inputting a window of 100 characters and training to generate the next character.
Thus from the dataset, subsets of window length 101 characters are generated and used as a single 
training example. The model is stateless so two different training windows are allowed to be overlapping,
just not identical.

## Modifications
1. If new datasets are found they should be placed in *datsets*. Then training or retraining can be done on that dataset.
2. The samples of generated text can be found in *results*.
3. Any text file can be used as dataset.
4. Currently it doesn't support using two or more textfiles as dataset.

## Logs and checkpoints structure
The logs and checkpoints are stored in a hierarchical fashion. The top level is grouping by dataset name.
Inside each such folder, the models and checkpoints are arranged by architecture. The label of such folder
is the system time when the model was first trained. Inside each such folder the logs are stored for all 
runs (the first one being training and all remaining being retraining). The lowest level folders are named
with date and time when the training/retraining was done. The same is the structure of checkpoints folder.

logs_<dataset>\
|----<system_time_for_architecture_1>\
|       |----<logs_for_train>\
|       |----<logs_for_retraining_1>\
|       |----<logs_for_retraining_2>\
|       ...\
|----<system_time_for_architecture_2>\
|       |----<logs_for_train>\
|       |----<logs_for_retraining_1>\
|       |----<logs_for_retraining_2>\
|       ...\
...\


# Available functions
The following functions can be currently done with the files present in the repository.


## Training new models
Some models are pretrained, however almost all the pretrained models are based on *sonnets* since
the architecture is large with a lot of parameters and the sonnets dataset is smaller. So new models for
the plays might be required. In order to train new models follow the folowing steps:

1. Write the model in *main.py* and keep everything else the same.
2. Execute the file in terminal or otherwise.
3. The name of the dataset file would be prompted. Enter the *filename*.
4. Enter the mode as *train*. The logging and checkpoints folder would be automatically generated.
5. Enter the number of *epochs*. Beware, training even a single epoch might take several hours.
6. After the execution completes, the logging directory would contain the logs and checkpoints would have the saved model.
7. Tensorboard can be used to visualise the models using the logs.


## Retraining previous models
As training is expensive it is advisable to run small number of epochs and just retrain the model.
In order to retrain the model follow the following steps:

1. Execute the *main.py* file in terminal or otherwise.
2. The name of the dataset file would be prompted. Enter the *filename*.
3. Enter the mode as *retrain*.
4. The entire directory listing for the particular dataset would be given. Choose the architecture to retrain.
5. The logs and checkpoints folder would be automatically generated.
6. Enter the number of *epochs*.
7. After the execution completes, the logging directory would contain the logs and checkpoints would have the saved model.
8. Tensorboard can be used to visualise the models using the logs.


## Generating novel text
After a particular architecture is trained to a sufficient degree, it can be used to generate text.
(Well even a model trained only once can be used to generate text, but results would be poor). In order
to generate shakespeare-like but novel text follow the following steps:

1. Execute the *generate.py* file in terminal or otherwise.
2. The name of the dataset file would be prompted. Enter the *filename*.
3. The entire directory listing for the particular dataset would be given. Choose the architecture to retrain.
4. A prompt will appear if the user wants to see the model summary.
5. Next a propmt for *context* would appear, this is what the model begins text generation with.
6. A propmt for number of characters would also be asked.
7. After text generation is complete, a prompt for saving the file would appear to store if result is satisfactory.
