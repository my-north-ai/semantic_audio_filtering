# TTS-data-augmentation
Synthetic data augmentation technique via LLM for audio. To serve as input for a TTS model training
![Explanation](/READMEIMAGE.png)

### The folders are distributed in the following order: 

configuratios -> Retrieve the basic settings from the .env file

data-> generation.ipynb has the pipeline for generating synthetic audio from multilingual libri speech dataset;
    generated_data-> storage for all the generated audio files

env.example -> example of how the .env file should be structured

