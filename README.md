# Whalify
It is an alexNet based binary classifier for detection of Orca/ non-Orca sounds using the spectrograms

## Spectrogram Generation
Orca Spectrogram    |  Non-Orca Spectrogram       
:-------------------------:|:-------------------------:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Whalify](dataset/test/1.png?raw=true "Whalify") &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;![Whalify](dataset/test/3.png?raw=true "Whalify")&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## Screenshots

The spectrograms are fed to the training model with set parameters. Below is the screenshot of training and validation loss and accuracy on different epochs.

![alttext](screenshots/train.png?raw=true "train")

After the model is trained, the final model is saved and finally inference is made on test set.

![Testing](screenshots/inference.png?raw=true "test")


### This is the edit by KJ
