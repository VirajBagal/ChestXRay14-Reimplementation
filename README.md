# ChestXRay14-Reimplementation
This is an attempt to reproduce the results of the paper: Jointly Learning Convolutional Representations to Compress Radiological Images and Classify Thoracic Diseases in the Compressed Domain, ICVGIP 2018, by Ekagra et al. 
Process is: 
  1. Train Stage 1. 
  2. Train Stage 2. 
  3. Train stage 3 using weights from stage 1 and stage 2.
  
```  
python train.py --datapath path_to_data --ckpt_path path_to_save_weights --stage stage_of_training
```
![alt text](https://github.com/VirajBagal/ChestXRay14-Reimplementation/master/images/model.png?raw=true)
