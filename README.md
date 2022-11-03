# Map the World with Sound

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

### input data

* Mel Spectrogram (頻譜圖)
* Raw data

### Output data

* 座標
* $\frac{d(座標)}{dt}$

### Model

* CRNN
    * 對頻譜圖做影像辨識
* Transformer
    * 直接把波形砸 in a nutshell


sound generation using: 
https://github.com/synthizer/synthizer

### HRTF 介紹

A head related transfer function (HRTF) describes the transformation of a specific source direction
relative to the head and filtering process associated with the diffraction of sound by
the pinna, head and torso.


