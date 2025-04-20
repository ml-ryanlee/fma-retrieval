import random
import librosa
import numpy as np
import audiomentations as am
from torch.utils.data import Dataset
# from datasets import load_dataset

class ContrastiveAudioDataset(Dataset):
    def __init__(self, dataset, sample_rate=44100):
        """
        Args:
            dataset: The dataset to be used (expecting free-music-archive-retrieval).
            sample_rate: The target sample rate.
        """
        self.dataset = dataset
        self.sample_rate = sample_rate

        self.audiomentations = am.Compose([
            am.AdjustDuration(duration_seconds=5.0, p=1),
            am.OneOf([
                # am.AddBackgroundNoise(p=1),
                am.Gain(min_gain_db=-10, max_gain_db=5, p=1),
                am.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=1),
                am.OneOf([
                    am.HighPassFilter(min_cutoff_freq=500, max_cutoff_freq=1000, p=1),
                    am.BandPassFilter(min_center_freq=500, max_center_freq=1000, p=1),
                    am.BandStopFilter(min_center_freq=500, max_center_freq=1000, p=1),
                    am.LowPassFilter(min_cutoff_freq=500, max_cutoff_freq=1000, p=1),
                ], p=1),
                am.PolarityInversion(p=1),
                am.TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
                am.TimeMask(min_band_part=0.1, max_band_part=0.2, p=1),
                am.PitchShift(min_semitones=-4, max_semitones=4, p=1),
            ], p=1,)
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]

            # Process original audio
            audio_data = sample["audio"]["array"]
            if(self.sample_rate != 44100):
                audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=self.sample_rate)

            if(random.random() < 1/8):
                # use existed q_audio_back for background noise
                transformed = librosa.resample(sample["q_audio_back"]["array"], orig_sr=44100, target_sr=self.sample_rate)
            else:
                # apply other transformation
                transformed = self.audiomentations(audio_data, sample_rate=self.sample_rate)

            return {
                "original": audio_data,
                "transformed": transformed,
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a valid sample with zeros if there's an error
            return {
                "original": np.zeros(5 * self.sample_rate),
                "transformed": np.zeros(5 * self.sample_rate),
            }

# if __name__ == "__main__":
    # fma_dataset = load_dataset("ryanleeme17/free-music-archive-retrieval", split="train")
    # dataset = ContrastiveAudioDataset(fma_dataset, 48000)