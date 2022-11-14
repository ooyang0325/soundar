# Map the World with Sound
<p align ="center"><img src="https://user-images.githubusercontent.com/52309935/201679137-983ad43a-ef6c-448d-9879-88063dc8ade5.png" width=50%></p>
<p align ="center"><img src="https://i.imgur.com/P1Ip1f9.jpg" width=75%></p>


## 分工

**楊詠翔**：

:heavy_check_mark: 美工(PTT製作等)、dataset 生成、~~瑟瑟~~、吸老婆、噁男擔當 

**詹挹辰**：

:heavy_check_mark: 研究文獻、Model Training

**葉宥辰**：

:heavy_check_mark: 研究文獻
發電、:zap:
:u5272: 
:ok_hand: 
:banana:

## input data

* Mel Spectrogram (頻譜圖)
* Raw data

## Output data

* 座標
* $\frac{d(座標)}{dt}$

## Model

* CRNN
    * 對頻譜圖做影像辨識
* Transformer
    * 直接把波形砸 in a nutshell
    * 目前的資料不適用(now we are using static single waveform file instead of a series of movement)


    sound generation using: 
    https://github.com/synthizer/synthizer

## HRTF 介紹

A head related transfer function (HRTF) describes the transformation of a specific source direction
relative to the head and filtering process associated with the diffraction of sound by
the pinna, head and torso.

## TO-DO list
- [x] pass the midterm(by any means) 
- [x] generate sound waveform files
- [x] create json file with waveform files' information
- [ ] build the enviroment on SYSTEX's server
- [ ] start training data
- [ ] marry my waifu **(!!important!!)** <br><br>
    > ooyang.waifu = {雷姆, 夕夕子, Aimyon, milet, 小松菜奈, 長澤茉里奈, 橋本環奈, 朝日奈まお, 檜山沙耶}


[^_^]:
    possible handcrafted features extraction: Mel-Frequency Cepstrum, skewness, kurtosis, log energy, entropy, zcr
    


