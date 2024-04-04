# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:57:07 2024

@author: Hp
"""

import torch
import torchaudio
from torch.nn.functional import cosine_similarity
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer, HubertModel
# Check if torchaudio's sox_io backend is available
if torchaudio.get_audio_backend() != 'sox_io':
    torchaudio.set_audio_backend("sox_io")


#%%
# Define the pre-trained models
models = {
    #'ecapa_tdnn': torch.hub.load('snakers4/silero-models', 'ecapa_tdnn'),
    'hubert_large': HubertModel.from_pretrained("facebook/hubert-large-ls960-ft"),
    'wav2vec2_xlsr': Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
}
#%%

# Function to preprocess audio clips
def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    TARGET_SAMPLE_RATE =16000
    # Resample if necessary
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Ensure single channel
    waveform = waveform.squeeze(0)  # Remove batch dimension if present
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Take the mean if multiple channels
    
    # Normalize waveform
    waveform /= torch.max(torch.abs(waveform))
    
    return waveform



# Function to extract embeddings from the models
def get_embedding(model, audio_clip):
    with torch.no_grad():
        embedding = model(audio_clip.unsqueeze(0))
    return embedding

# Function to perform speaker verification
def speaker_verification(reference_clip, test_clip, model_name):
    # Load the pre-trained model
    model = models[model_name].eval()
    
    # Preprocess audio clips
    ref_audio = preprocess_audio(reference_clip)
    test_audio = preprocess_audio(test_clip)
    
    # Extract embeddings
    with torch.no_grad():
        ref_output = model(ref_audio.unsqueeze(0)).last_hidden_state
        test_output = model(test_audio.unsqueeze(0)).last_hidden_state
    
    # Flatten the embeddings
    ref_embedding = ref_output.squeeze(0)
    test_embedding = test_output.squeeze(0)
    
    # Ensure the dimensions match
    min_length = min(ref_embedding.shape[0], test_embedding.shape[0])
    ref_embedding = ref_embedding[:min_length]
    test_embedding = test_embedding[:min_length]
    
    # Calculate cosine similarity
    similarity = cosine_similarity(ref_embedding, test_embedding, dim=1)
    
    return similarity.mean().item()  # Taking mean of similarity scores and then converting to scalar


#%%

# Example usage
reference_clip = 'reference_clip.wav'
test_clip = 'test_clip.wav'
model_name = 'wav2vec2_xlsr' #'hubert_large'  #'wav2vec2_xlsr'  
#%%
#waveform, _ = torchaudio.load(reference_clip)

#%%
similarity_score = speaker_verification(reference_clip, test_clip, model_name)
print(f"Similarity Score ({model_name}): {similarity_score}")
