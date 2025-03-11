<p align ="center"><img src="https://user-images.githubusercontent.com/52309935/201679137-983ad43a-ef6c-448d-9879-88063dc8ade5.png" width=15%></p>
<h1 align="center"> Soundar - Map the World with Sound</h1>

We use MLP, Polynomial Regression Model to predict DOA (Direction of Arrival) of binaural audio tracks, and use Lasso Regression Model to predict the distance of audio source.

## Features

1. Binaural localization mode, replacing traditional multi-microphone array localization
2. Easier to apply to wearable devices (headphones, hearing aids, etc.)
3. High accuracy in angle prediction
​
## Application

1. Assist hearing-impaired individuals in noticing potential sudden threats
2. Aid in detecting mechanical failures in automated production lines
3. Auditory systems for bionic robots/animal robots
4. Enhance surveillance capabilities in security systems
​
## Dataset

The dataset consists of six types of sound sources: sine waves (130.81 Hz, 261.63 Hz, 1046.5 Hz), ambulance noise, gunshot, fart. Dataset is generated with [synthizer](https://github.com/synthizer/synthizer).

* Dataset format:
  * Position (x, y): Sound source cartesian coordinates.
  * Audio (stereo): Two-channel recording (left and right ear) at the listener's position (0,0).
​
## ML Models

### DOA (Direction of Arrival) Prediction

* Features: ITD (Interaural Time Difference), ILD (Interaural Level Difference)
* Output: DOA (incoming sound angle)
* Selected Models: MLP (classification), Polynomial Regression

### Distance Prediction

* Feature: DOA, ITD, ILD, RMS Energy
* Output: R (distance)
* Selected Model：Lasso Regression Model (L1 regularization)

<p align="center">
<img src="https://i.imgur.com/zOJPF6U.png" width=60%>
<br>
ITD and ILD data distribution, different colors represent different angles.
</p>

## Results

<p align="center">
<img src="https://i.imgur.com/S2SsPU7.jpg" width=60%>
<br>
MLP classification results for DOA.
</p>

<br>

<p align="center">
<img src="https://i.imgur.com/Ad7ui76.jpg" width=60%>
<br>
Polynomial regression results for DOA.
</p>
<br>

<p align="center">
<img src="https://i.imgur.com/g2Rnc1u.jpg" width=60%>
<br>
Lasso regression results for Cartesian coordinate.
</p>
