<p align ="center"><img src="https://user-images.githubusercontent.com/52309935/201679137-983ad43a-ef6c-448d-9879-88063dc8ade5.png" width=15%></p>
<h1 align="center"> Soundar - Map the World with Sound</h1>

使用 MLP, Polynomial Regression Model 等多種模型進行 DOA (音源方位角度) 的預測，並且使用 Lasso Regression Model 進行距離的預測。


## 專題特色
1. 雙聲道定位模式，取代以往使用多麥克風陣列定位方式
2. 更容易應用於貼身裝置(耳機、助聽器等)
3. 預測角度的準確率高
​
## Application
1. **協助失聰人士注意潛在的突發威脅**
2. 在自動化產線中輔助偵測機械故障
3. 仿生機器人/機器動物的聽覺系統
4. 應用於保全系統中提供更完善的監控能力
​
## Dataset
* 總共使用六種音源所生成的 dataset: sin wave of 130.81, 261.63, 1046.5, ambulance noice, gunshot, fart
* 每個 dataset 的形式
    * $R=1\sim 30$, 公差 $0.5$
    * $degree=0,5,10,15,\cdots,175,180$
​
## 機器學習模型

### DOA 預測

* Feature: ITD, ILD
* Ouput: DOA
* 選用模型：MLP, Polynomial Regression, GMM(用於檢驗)

### 距離預測

* Feature: DOA, ITD, ILD, 方均根能量
* Output: R (距離)
* 選用模型：lasso

<p align="center">
<img src="https://i.imgur.com/zOJPF6U.png" width=60%>
<br>
ITD, ILD 數據分佈圖，不同顏色為不同角度
</p>

## Result

<p align="center">
<img src="https://i.imgur.com/S2SsPU7.jpg" width=60%>
<br>
MLP 預測 DOA 結果展現
</p>

<br>

<p align="center">
<img src="https://i.imgur.com/Ad7ui76.jpg" width=60%>
<br>
Polynomial Regression 預測 DOA 結果展現
</p>
<br>

<p align="center">
<img src="https://i.imgur.com/g2Rnc1u.jpg" width=60%>
<br>
直角坐標預測結果
</p>
