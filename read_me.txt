Hello my friends, this is a pre-view version, there might be some bugs.


 If you have any question, feel free to drop emails to barid.x.ai@gmail.com.
 I am happy to answer any question.


Hints.
1. LPA (the proposed model) is in core_resTransformer.py .
2. The whole lip reading system is in core_lip_main.py .
3. Lip detector is in lip_detection fold.


@dataset link:
http://www.robots.ox.ac.uk/~vgg/data/
After download, you need to use sentence_tfrecoder_generator.py for data preprocessing.
python sentence_tfrecoder_generator.py -t "path to data index" -v "path to lip reading data"
for "path to data index", LRS3&2 list the video indexes for pre-training and training.

Note that:
1. this is not necessary. You can directly pass the data to the model. However, we recommend
this because TFRecord has faster loading speed.

2. please download shape_predictor_68_face_landmarks.dat for dlib: https://github.com/davisking/dlib-models 
and then save to pre_train


@Sample difficulty:
To change the sample difficulty, you can change the degree in core_data_SRCandTGT.py.

@Usage:
python main -m 'LIP'
opitons:
-m 'LIP', 'GPT' and 'VIS' ('LIP' for ResSA pre-training and formal training; 'GPT' for decoder pre-training, 'VIS' for visual module pre-training)
