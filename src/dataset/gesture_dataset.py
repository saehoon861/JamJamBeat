import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob

class GestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_to_id = {}
        self.id_to_label = []

        self._load_data()

    def _load_data(self):
        csv_files = glob(os.path.join(self.data_dir, "*.csv"))
        
        all_frames = []
        all_frame_labels = []

        # First pass to build label_to_id mapping
        unique_labels = set()
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            # Assuming filename format: NUMBER_LABEL_...
            # e.g., '0_neutral_right_man1.csv' -> 'neutral'
            label_str = file_name.split('_')[1]
            unique_labels.add(label_str)
        
        self.id_to_label = sorted(list(unique_labels))
        self.label_to_id = {label: i for i, label in enumerate(self.id_to_label)}

        for file_path in csv_files:
            df = pd.read_csv(file_path)
            # Assuming all columns except the first (if it's an index/timestamp) are landmark data
            # And landmark data corresponds to 63 features
            # I will assume the CSV directly contains 63 columns of landmark data, no timestamp
            # If there's a timestamp, it's usually the first column, so I'll drop it if df.shape[1] > 63
            
            # Extract label from filename
            file_name = os.path.basename(file_path)
            label_str = file_name.split('_')[1]
            label_id = self.label_to_id[label_str]

            # Convert DataFrame to numpy array, then to torch tensor
            # Ensure the data has 63 features. If more, assume first column is index and drop it.
            if df.shape[1] > 63:
                # Assuming the first column is an identifier/timestamp
                landmarks = df.iloc[:, 1:].values # take all rows, from second column onwards
            else:
                landmarks = df.values # take all columns

            # Convert to float and store frames and their labels
            for frame_data in landmarks:
                # Ensure each frame has 63 features
                if len(frame_data) == 63:
                    all_frames.append(torch.tensor(frame_data, dtype=torch.float32))
                    all_frame_labels.append(torch.tensor(label_id, dtype=torch.long))
                else:
                    print(f"Warning: Skipping frame from {file_path} with {len(frame_data)} features (expected 63).")

        self.data = all_frames
        self.labels = all_frame_labels
        print(f"Loaded {len(self.data)} total frames from {len(csv_files)} files.")
        print(f"Detected {len(self.id_to_label)} classes: {self.id_to_label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            frame = self.transform(frame)
        
        return frame, label
