import sys
import os
sys.path.insert(0, '/mnt/d/hermes_playground/Praasper')

from praasper import init_model

# Initialize the model
model = init_model()

# Run annotation on the full directory.
# By NOT passing params, it triggers the embedded auto_vad() grid search
# to find the optimal parameters for each file automatically.
model.annote(
    input_path='/mnt/d/audio_data',
    skip_existing=False, # Run on all files
    verbose=False # Less verbose to save log space, show progress bars
)
