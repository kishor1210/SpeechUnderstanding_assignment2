# SpeechUnderstanding_assignment2
Question 1. Speaker Verification Goal: In speaker verification, the training dataset consists of audio clips paired with speaker IDs, denoted as (D = (xi , yi)). Given an audio clip (x) and a reference clip (x0), the objective is to ascertain whether (x0) and (x) belong to the same speaker. 

Tasks: —- Choose three pre-trained models from the list: ’ecapa tdnn’, ’hubert large’, ’wav2vec2 xlsr’,
’unispeech sat’, ’wavlm base plus’, ’wavlm large’ trained on the VoxCeleb1 dataset. You can find the
pre-trained models on this link.
—- Calculate the EER(%) on the VoxCeleb1-H dataset using the above selected models. You can get the
dataset from here. [3 Marks]
—- Compare your result with Table II of the WavLM paper. [2 Marks]
—- Evaluate the selected models on the test set of any one Indian language of the Kathbath Dataset.
Report the EER(%). [2 Marks]
—- Fine-tune, the best model on the validation set of the selected language of Kathbath Dataset. Report
the EER(%). [10 Marks]
—- Provide an analysis of the results along with plausible reasons for the observed outcomes. [5 Marks]
