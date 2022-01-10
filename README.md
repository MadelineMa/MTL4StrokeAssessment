# Multi-objective learning for stroke risk assessment

This repo provide the implementation of the stroke risk assessment. Three main models are considered, DNN, QIDeep and MMOE with the stroke occurrence prediction as the auxiliary task. We briefly introduced the features and the models. Due to the privacy issue, the stroke assessment dataset is not released. However, we apply those models to the MNIST handwritten dataset.

## Features
   
| 中文 | 英文 | 缩写 | 数量 |
|:---:| :---: | :---:| :---: |
| 性别 |  Gender  |  Gd| 2 |
| 建档年龄 | Filling Age | FA | 77 |
| 民族 | Nationality | Nat | 10 |
| 婚姻状况 | Marital Status | MS | 6 |
| 受教育程度 | Education | Edu | 7 |
| 是否退休 | Retirement | Ret | 3 |
| 风险评级 (Label) | Risk Rating | RR | 4 |
| 卒中 (Label)| Apoplexy | Apx |
| 吸烟 | Smoke | Sm | 3 |
| 吸烟年限 | Years of Smoke | YS | 60 |
| 饮酒 | Drink | Drk | 3 |
| 缺乏运动 | Exercise | Exs | 2 | 
| 口味 | Flavor | Flv | 3 | 
| 荤素 | Meat &  Vegetable | MV | 3 | 
| 食用蔬菜 | Vegetable Comsumption | VC| 3 |
| 食用水果 | Fruit Consumption | FC | 3 |
| 脑卒中 | History of Apoplexy | HA | 3 |
| 冠心病 | History of Coronary Artery Disease | HCAD | 3| 
| 高血压 | History of Hypertension | HEH | 3 |
| 糖尿病 | History of Diabetes Mellitus | HDM | 3 |
| 身高 | Height | Ht | 373 |
| 体重 | Weight | Wt | 567 |
| BMI |   |   | 1705
| 左侧收缩压 | Left Systolic Blood Pressure | LSBP | 146 |
| 左侧舒张压 | Left Diastolic Blood Pressure | LDBP | 105 |
| 右侧收缩压 | Right Systolic Blood Pressure | RSBP | 148 |
| 右侧舒张压 | Right Diastolic Blood Pressure | RDBP | 95 |
| 心律 | Rhythm | Rhm | 3 |
| 脉搏 | Pulse | pls | 87 |
| 空腹血糖 | Fasting Blood-glucose | FBG | 623 |
| 糖化血红蛋白 | Glycosylated Homoglobin | HbA1c | 141 |
| 甘油三脂 |  | TG | 549 |
| 总胆固醇 |  | TC | 635 |
| 低密度脂蛋白胆固醇 |   | LDL-C | 537 |
| 高密度脂蛋白胆固醇 |   | HDL-C | 309 |
| 同型半胱氨酸 | Homocysteine | Hcy | 1318 |
 
## Models
### baseline
We considered baseline models in the baseline directory, which invoke `sklearn` and `keras`.

### src  

We write the original codes with PyTorch in this directory, which contains:
   
<!-- - `preStroked`: `卒中.xlsx` $\to$ `2019-stroked.csv`.   -->
<!-- - `preprocess`: `基础信息.xlsx` + `档案信息.xlsx` $\to$ `2019-merged.csv`.    -->
- `strokeData`: `preStroked` + `preprocess` -> `normalStroke.csv` -> `train.csv` + `valid.csv` + `test.csv`.   
- `strokeDataset`:   
	- `StrokedDataset`: 风险评级 +卒中 -> data + stroke\_labels + risk\_labels.   
	- `DnnDataset`: RR -> data + labels.   
	- `QIDataset`: RR -> data, index, labels.   
	- `BinaryDataset`: 卒中 -> data, labels.   
	-  `MoEDataset`: RR + Apx -> data + index + slabels + rlabels.   
- `strokeModels`: Stroke\_LR, Risk\_DNN, StrokeRistModel. (Tips: nn fixed!).  
- `strokeUtils`: template of loss\_batch and fit
<!-- - `logistic_regression`: **TODO** plot.      -->
- `strokeMain`: $l_s + l_r + l_{loc}$.

#### dnn  

- `dnn_binary`: 
- `dnn_data`: `normalStroke.csv` -> `dnn_train.csv` + `dnn_valid.csv` + `dnn_test.csv`.   
- `dnn_risk`: implementation of stroke risk assessment with 34 input features.
- `dnn_risk_sf`: implementation of stroke risk assessment with top 20 input features.   
- `dnn_loop_risk`: statistical study of the stroke risk assessment.
- `dnn_loop_binary`: statistical study of the stroke occurrence prediction.

#### QI

- `QI_data`: `normalStroke.csv` -> `dnn_train.csv` + `dnn_valid.csv` + `dnn_test.csv` + `dnn_*_sf.csv` + `dnn_*_ind.csv` + `normalCategory.csv`.   
<!-- - `QI_emb`: embedding only, no plot.    -->
- `QI_emb_Category`: QI values are constructed "piecewisely".
<!-- - `QI_emb_data`:  -->
<!-- - `QI_emb_input`: appending QI values to input, no plot, feature\_sizes fixed.       -->
<!-- - `QI_emb_linear`: no plot, feature\_sizes fixed.    -->
<!-- - `QI_embx`:  $v_ix_i$, no plot, feature\_sizes fixed.    -->
- `QI_risk`:  $v_ix_i$, no plot, feature\_sizes fixed.    
- `category_data`: `normalCategory.csv` + `./qi/`.   

#### MoE 

- `MoE`:
- `MoE_data`:  
 
#### MNIST
Codes of the above methods applied to the MNIST handwritten digits.

## Citation
If you are using the code, please cite the following work
```
@article{JingMa2021,
  author    = {Jing Ma and Yiyang Sun and Junjie Liu and Huaxiong Huang and Xiaoshuang Zhou and Shixin Xu},
  title     = {Multi-objective optimization and explanation for stroke risk assessment in Shanxi province},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.14060},
  eprinttype = {arXiv},
  eprint    = {2107.14060},
  timestamp = {Tue, 03 Aug 2021 14:53:34 +0200},
}
```