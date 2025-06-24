# Emotion classification on speech data

> ⚠️ **Caution**  
>  
> I discussed the issue with Arpit from the Mars Club regarding my project model, which is approximately **3.1 GB** in size and could not be pushed to GitHub due to file size limitations.  
> Based on his suggestion, I have included the following components in the repository:  
>  
> - **`Notebook.ipynb`**: Contains the core experiments along with corresponding plots and graphs.  
> - **`notebook_weighted_training.ipynb`**: Includes experiments using weighted training to improve precision for classes that were initially below 75%.  
> - **`Demo Video.mp4`**: A demonstration video showcasing the model's functionality and performance.  
>  
> Further details regarding project execution can be found in the **Project Execution** section of this README.



## Project Overview

This project focuses on **emotion classification** of various audio samples—specifically **speech and song segments**—into the following 8 categories:

> ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
> 

The classification model used is **`facebook/wav2vec2-large`**.

### Model Training

- The dataset was split into **80/20 for training and validation**.
- Initial fine-tuning was performed on this split, but the precision for the `sad` and `disgust` classes was observed to be **below 75%**.
- To address this, **weighted training** was applied:
    - `sad`: weight = 3
    - `disgust`: weight = 2
    - All other classes: weight = 1

### Preprocessing

- The **feature extractor** from `facebook/wav2vec2-base` was used as the preprocessor.
- Multiple preprocessing techniques were evaluated, but this configuration yielded the best results.

### Evaluation & Results

- As the final model could not be uploaded (due to size limitations), evaluation metrics are provided on the validation data in the following files:
    - **`test.ipynb`**: Includes confusion matrix, overall accuracy, per-class precision, and F1 score.
    - **`test.py`**: Provides equivalent evaluation in script format. Instructions for running this script on a folder of audio files are provided in the code comments.
- Additionally, outputs from model training (including logs and cell outputs) are retained in the notebook for verification.

### Streamlit App

A simple **Streamlit app** is included that accepts `.wav` files as input and outputs the predicted emotion classification.

---

## Project Execution

```bash
# Clone the repository
git clone https://github.com/37nomad/Speech-Classification.git

# Navigate to project directory
cd Speech-Classification

# For normal training and loss plots:
# Open the Jupyter notebook
open Notebook.ipynb

# For weighted training:
open notebook_weighted_training.ipynb

# To test the model (requires a folder of audio files):
python3 -u test.py

# To launch the Streamlit app:
streamlit run app.py
```
