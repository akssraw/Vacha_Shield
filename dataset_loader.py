import os
import torch
import numpy as np
from torch.utils.data import Dataset
from feature_extraction import extract_features

class ASVspoof2019Loader(Dataset):
    """
    A custom PyTorch Dataset that parses the official ASVspoof 2019 Logical Access (LA) database.
    
    The LA dataset has protocol files (text files) that map the randomly named audio files
    (e.g., LA_T_1138215) to their true label ('bonafide' or 'spoof').
    """
    def __init__(self, dataset_path, protocol_file, subset='train', feature_type='spectrogram', max_pad_len=400):
        """
        Args:
            dataset_path (str): Path to the base of the ASVspoof 2019 LA dataset, 
                                e.g., 'path/to/ASVspoof2019_LA_train'
            protocol_file (str): Path to the protocol list, 
                                 e.g., 'path/to/ASVspoof2019.LA.cm.train.trn.txt'
            subset (str): Usually 'train', 'dev', or 'eval' (determines the flac folder)
            feature_type (str): 'spectrogram' or 'mfcc'
        """
        self.dataset_path = dataset_path
        self.protocol_file = protocol_file
        self.feature_type = feature_type
        self.max_pad_len = max_pad_len
        
        # Determine the audio directory (FLAC files) based on subset
        self.audio_dir = os.path.join(self.dataset_path, 'flac')
        
        self.data_list = [] # Will hold tuples of (absolute_file_path, label_num)
        
        print(f"Loading ASVspoof 2019 LA Protocol: {protocol_file}...")
        self._parse_protocol()
        print(f"Successfully indexed {len(self.data_list)} audio samples.")

    def _parse_protocol(self):
        """
        Reads the ASVspoof protocol text file line by line.
        Format of each line:
        SPEAKER_ID AUDIO_FILE_ID SYSTEM_ID METHOD KEY
        Example:
        LA_0079 LA_T_1138215 - - spoof
        LA_0079 LA_T_1271820 - - bonafide
        """
        if not os.path.exists(self.protocol_file):
            raise FileNotFoundError(f"Protocol file not found at: {self.protocol_file}")
            
        with open(self.protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            # ASVspoof 2019 LA format has 5 columns
            if len(parts) >= 5:
                audio_id = parts[1] # e.g., LA_T_1138215
                key = parts[4]      # e.g., spoof or bonafide
                
                # We map 'bonafide' (Human) to 0.0 and 'spoof' (AI/Deepfake) to 1.0
                label = 0.0 if key == 'bonafide' else 1.0
                
                # Construct exact path to the .flac file
                file_path = os.path.join(self.audio_dir, f"{audio_id}.flac")
                
                if os.path.exists(file_path):
                    self.data_list.append((file_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        
        # 1. Extract the spectrogram or MFCC using our engine
        features = extract_features(file_path, max_pad_len=self.max_pad_len, feature_type=self.feature_type)
        
        # 2. Handle parsing errors securely
        if features is None:
            channels = 1 if self.feature_type == "mfcc" else 2
            features = np.zeros((channels, 40, self.max_pad_len), dtype=np.float32)
            
        # features shape: (2, 40, 400) — dual channel from upgraded extract_features
        # No unsqueeze needed; channel dim is already present
        tensor_features = torch.tensor(features, dtype=torch.float32)
        tensor_label = torch.tensor([label], dtype=torch.float32)
        
        return tensor_features, tensor_label

if __name__ == "__main__":
    print("ASVspoof Loader Ready.")
    # Example usage:
    # dataset = ASVspoof2019Loader(
    #     dataset_path="data/LA/ASVspoof2019_LA_train",
    #     protocol_file="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    # )
    # print(f"Length of dataset: {len(dataset)}")
