# DeepLearning-Big-Project 
50.039 Deep Learning, Y2021   
Group Member: Zachary, Leng Sicong, Li Jiaxi 

## Dataset and pre-trained models preparation
Please download dataset and pre-trained models zip file from [Google Drive](https://drive.google.com/file/d/1WfPSN2dEiHR1L6tMImkUX6FAOu0FCXgp/view?usp=sharing)  
Then unzip the file under the root directory:
```
unzip Data_and_Model.zip
```
## Training and Testing
###### Charades-STA dataset
To train the model from scratch, run the following commands:
```
python main_predictor.py --visual_dim 1024 --model_dir saved_char/ --task charades  --pos_freeze --vq_mi --vq_mi_lambda 0.3 --vv_mi --vv_mi_lambda 0.3 --mode train
```
To test existing pre-trained model and reproduce the result:
```
python main_predictor.py --visual_dim 1024 --model_dir saved_char/ --task charades  --pos_freeze --vq_mi --vq_mi_lambda 0.3 --vv_mi --vv_mi_lambda 0.3 --mode test
```
###### TACoS dataset
To train the model from scratch, run the following commands:
```
python main_predictor.py --model_dir saved_tacos/ --task tacos  --pos_freeze --vq_mi --vq_mi_lambda 0.3 --vv_mi --vv_mi_lambda 0.3 --mode train
```
To test existing pre-trained model and reproduce the result:
```
python main_predictor.py --model_dir saved_tacos/ --task tacos  --pos_freeze --vq_mi --vq_mi_lambda 0.3 --vv_mi --vv_mi_lambda 0.3 --mode test
```
## Report
Please refer to [PDF](./README.md) or [Google Doc](https://docs.google.com/document/d/1KQQfK6iFr2tw78_a8SbihfnOFV-jIkp0Gfviat-rM80/edit?usp=sharing) for more details. 
