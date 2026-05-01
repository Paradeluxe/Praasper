import sys
import os
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')

from praasper import init_model

# Initialize the model
model = init_model()

# Run annotation on the full directory using the NEW default parameters
# params=None would trigger auto_vad (fine-tuning). 
# Passing default_params uses the fixed values we just optimized.
model.annote(
    input_path='/mnt/d/audio_data',
    params=model.default_params,
    skip_existing=False, # Run on all files
    verbose=True
)
