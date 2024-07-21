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

& The dataset consists of six types of sound sources: sine wave of 130.81, 261.63, 1046.5, ambulance noise, gunshot, fart
* Each dataset format:
   * $R=1\sim 30$, with a tolerance of $0.5$
   * $degree=0,5,10,15,\cdots,175,180$
​
## ML Models

### DOA Prediction

* Features: ITD, ILD
* Ouput: DOA
* Selected Models: MLP, Polynomial Regression, GMM (for validation)

### Distance Prediction

* Feature: DOA, ITD, ILD, RMS Energy
* Output: R (distance)
* Selected Model：Lasso Regression Model

<p align="center">
<img src="https://i.imgur.com/zOJPF6U.png" width=60%>
<br>
ITD and ILD data distribution, different colors represent different angles
</p>

## Results

<p align="center">
<img src="https://i.imgur.com/S2SsPU7.jpg" width=60%>
<br>
MLP prediction results for DOA
</p>

<br>

<p align="center">
<img src="https://i.imgur.com/Ad7ui76.jpg" width=60%>
<br>
Polynomial Regression prediction results for DOA
</p>
<br>

<p align="center">
<img src="https://i.imgur.com/g2Rnc1u.jpg" width=60%>
<br>
Cartesian coordinate prediction results
</p>
