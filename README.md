# Sign Language Translation

This project explores the use of Convolutional Recurrent Neural Networks to translate a curated subset of American Sign Language (ASL) gestures directly into text through a live webcam feed.

Using the WLASL video dataset, alongside MediaPipe for precise hand-landmark and pose detection, members will learn how to transform videos of gestures into meaningful features suitable for machine learning. Early milestones include classifying single frames of hand gestures, before transitioning into recognition of gestures that consist of spatial changes over multiple frames using Long Short-Term Memory (LSTM) layers. The project culminates in a Streamlit application that visualizes hand movements and skeletal structures on screen in real-time, and displays translations using the trained gesture recognition model.

**Final deliverable:** A streamlit dashboard with an interface that provides real-time sign language translation, with a visualization of the skeletal structure and display of both the ASL glosses and the english translation of multiple gestures in sequence


# Week 1
- [Week 1 Slides](https://docs.google.com/presentation/d/1mJ6QjMuurNHb7etnTNe96_rq9khBtgGW8bxEydZ8n68/edit?usp=sharing)
- [Week 1 Notebook - Intro to MediaPipe](https://colab.research.google.com/drive/1bkYvOFaU2PBsfJxAZBg9k0pSEUVr8SAu?usp=drive_link)
- Download image: `!wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg`

# Week 2
- [Week 2 Slides](https://docs.google.com/presentation/d/1SBArJkI8JzcSzYju07fL3S4_uXxj5jDPHdy_tK8KAic/edit?slide=id.g3bc92866f7b_2_300#slide=id.g3bc92866f7b_2_300)

# Week 3
- [Week 3 Slides](https://docs.google.com/presentation/d/1AVtqcNsD8Uw9kVlnnCw_Dvl9rETAWDoTYzptZpswKXc/edit?usp=drive_link)

# Week 4
- [Week 4 Reading](https://www.handspeak.com/learn/98/)
- [Week 4 Slides](https://docs.google.com/presentation/d/1z2I8vZRb1QiWqPLZIYbeTonWW0ZZw4xAj_oPC-LuPV8/edit?usp=sharing)
- [Week 4 Notebook](https://colab.research.google.com/github/MichiganDataScienceTeam/W26-Sign-Language-Translator/blob/main/Week%204/week4_wlasl.ipynb)
- Dataset and starter code: [wlasl_week4.zip](https://drive.google.com/file/d/15l1LFlF9gTqms8ufQYqA1ry_qoWlOXVL/view?usp=sharing)
- [Form for sharing results](https://docs.google.com/forms/d/e/1FAIpQLSell28hoohp7f9UpNQgXAbUgE-dG5ll6G1uaZ5dlusEZlQRJg/viewform?usp=sharing&ouid=113884584305464810417)

# Week 5
- [asl_glosses.zip](https://drive.google.com/file/d/1YQHooWEJyw-4aTZhRpFPN9TOH95Xj61w/view?usp=sharing)

# Google Colab Setup Guide

Start by uploading your notebook to Google [Colab](https://colab.research.google.com/). You can also access the notebook through Colab via the following link:

<a href="https://colab.research.google.com/github/MichiganDataScienceTeam/W26-Sign-Language-Translator/blob/main/Week%204/week4_wlasl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

On the top left of your screen, click "File > Save as copy to Drive" or click "Copy to Drive" so that you can save your changes.


To enable the GPU on Google Colab, select the following settings from the menu bar at the top:
```python
# colab: enable GPU:
# Runtime > Change Runtime Type > T4 GPU > Save

# note: may need to restart your Notebook session (Runtime > Restart Session)
```

For running on Google Colab, note that you won't be able to use a web camera for real-time classification, however you can use Colab to train your model using the provided GPU option, then download the saved model weights to your local machine and run inference using your computer's CPU. See the [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) or feel free to ask us if you need assistance with saving and loading model weights.

```python
model = LSTM_Model()
model.to(device)
model.train()
# ... train your model ...
torch.save(model.state_dict(), "model_weights.pth")

# <download from Colab to your local setup>

# load model weights using same class definition used for training (LSTM_Model)
model = LSTM_Model()
model.load_state_dict(torch.load("model_weights.pth"))
model.to(device)
model.eval()
```