
import sys
sys.path.insert(0, r'D:\hermes_playground\Praasper')
from praasper import init_model

print("Initializing model...")
model = init_model(device='cpu')  # force cpu to avoid vram conflicts, or auto

print("Running auto_vad on 01-1...")
model.annote(
    input_path=r'D:\audio_data\01-1.wav',
    verbose=True,
    skip_existing=False
)
print("Done.")
