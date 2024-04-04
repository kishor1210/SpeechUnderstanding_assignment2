import torch
import torch.nn as nn
import torchaudio
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer, HubertModel
# Check if torchaudio's sox_io backend is available
if torchaudio.get_audio_backend() != 'sox_io':
    torchaudio.set_audio_backend("sox_io")
#%%
# Step 1: Define a dataset class to load the data
class VoxCelebDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [line.split() for line in file.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Step 2: Define your model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load pre-trained model and tokenizer
        #model_name = "facebook/wav2vec2-large-xlsr-53"
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        #self.model = Wav2Vec2Model.from_pretrained(model_name).eval()

    def forward(self, x1, x2):
        # Assuming x1 and x2 are paths to audio files
        # You need to implement how to load and process audio files into features
        # Convert audio files to features
        feature1 = self.preprocess_audio(x1)
        feature2 = self.preprocess_audio(x2)
        
        # Extract embeddings
        with torch.no_grad():
            out1 = self.model(feature1.unsqueeze(0)).last_hidden_state
            out2 = self.model(feature2.unsqueeze(0)).last_hidden_state
        
        # Flatten the embeddings
        out1_emb = out1.squeeze(0)
        out2_emb = out2.squeeze(0)
        
        # Ensure the dimensions match
        min_length = min(out1_emb.shape[0], out2_emb.shape[0])
        output1 = out1_emb[:min_length]
        output2 = out2_emb[:min_length]


        # Here, you need to define how to compute similarity between output1 and output2
        # For example, you can use cosine similarity, Euclidean distance, etc.
        similarity_score = self.compute_similarity(output1, output2)

        return similarity_score
    
    
    # Function to preprocess audio clips
    def preprocess_audio(self, audio_path):
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


    def compute_similarity(self, output1, output2):
        # Implement similarity computation here
        # For demonstration, let's assume we compute cosine similarity between output1 and output2
        # You might need to reshape or process the outputs before computing similarity
        # Here's a simple example of computing cosine similarity
        # Note: This is just a placeholder. Implement the actual similarity computation as needed.
        # Calculate cosine similarity
        similarity = cosine_similarity(output1, output2, dim=1)
        return similarity.mean().item()

# Step 3: Calculate EER
def calculate_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100  # Convert to percentage
#%%
# Load the dataset
dataset = VoxCelebDataset("datasets/list_test.txt")

# Initialize the model
model = MyModel()

# Load pre-trained weights if necessary
# model.load_state_dict(torch.load('pretrained_model.pth'))

# Lists to store scores and labels
scores = []     
labels = []
no_file = []

#%%
#score = model("reference_clip.wav", "test_clip.wav")
#score

#%%
# Iterate through the dataset and generate scores
for data in dataset:
    # Here, data[0] and data[1] represent the paths of two audio files for comparison
    # You need to implement how you load and process these audio files
    # Then, pass them through your model to get the similarity score
    try:
        score = model('datasets/wav_007/'+data[1], 'datasets/wav_007/'+data[2])  # Adjust this line according to your model's input
        print(score)
        scores.append(score)
        labels.append(int(data[0]))
        print(data[1],data[2])
    except:
        no_file.append(data)

#%%
# Calculate EER
eer = calculate_eer(scores, labels)
print(f"EER: {eer:.2f}%")
