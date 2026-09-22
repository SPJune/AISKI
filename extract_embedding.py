from speechbrain.pretrained import SpeakerRecognition
import librosa
from glob import glob
import torch
import numpy as np
from pathlib import Path
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_ecapa")

precue_dir = Path("outputs/precue")
embedding_dir = precue_dir / "embeddings"
embedding_dir.mkdir(parents=True, exist_ok=True)

wav_list = glob(str(precue_dir / "*.wav"))
for wav_path in wav_list:
    x, sr = librosa.load(wav_path, sr=16000)
    name = str(embedding_dir / (Path(wav_path).stem + ".npy"))
    embedding = verification.encode_batch(torch.from_numpy(x))
    embedding = embedding.squeeze().numpy()
    print(name)
    np.save(name, embedding)
