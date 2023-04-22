# OLIVES_Classification_ECE8803

Final paper and pptx with audio recording is included. I cannot upload the mp4 of the presentation here because it is too big. The voiceover for each slide is on the left of each slide. A zip of the mp4 was included on the Canvas submission, you can get it there or export from the pptx file here yourself.

Each of the training files can be run individualy in their current state. However if you want to run all 4 training files just run the main.py file. Make sure that the dataset is local to the folder the repo is cloned into. I put the Prime_FULL folder here at least to let you know that the dataset should be contained within there.

The finished models are not able to be uploaded due to their file size and github restrictions. If you run the training loop now the results will most likely be slightly different than in the project. That's because the project one's are a result of learning on strictly augmented data and then strictly non augmented data. The way as it trains now will only be on non-augmented data. It'll actually get to a slightly better solution than on just augmented data, however it will generalize pretty poorly.

The current training epochs are set to 250 for the large models and 50 for the CAE. That should be good for where the paper is at, though truthfully for decent results probably need a couple thousand.
