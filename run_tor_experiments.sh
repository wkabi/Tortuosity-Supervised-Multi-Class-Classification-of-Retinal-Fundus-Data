#!/usr/bin/env bash

## TOR training

## Bit models
#python train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext101_1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_bit_resnext101_1_sgd_cy10by3_4cls

#python train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_bit_resnext50_1_sgd_cy10by3_4cls

#python train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1_KD --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_bit_resnext50_1_KD_sgd_cy10by3_4cls

#python train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1_MOD --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_bit_resnext50_1_MOD_sgd_cy10by3_4cls

## MobileNet models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name mobilenetV2 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_mobilenetV2_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name mobilenetv3_large_100 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path Tor_mobilenetv3_large_100_sgd_cy10by3_4cls

## ResNet models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnet18 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnet18_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnet34 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnet34_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnet50 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnet50_sgd_cy10by3_4cls

## EfficientNet models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name efficientnet_b5 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_efficientnet_b5_sgd_cy10by3_4cls

##Error:python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name efficientnet_b6 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_efficientnet_b6_sgd_cy10by3_4cls

##Error:python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name efficientnet_b7 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_efficientnet_b7_sgd_cy10by3_4cls

## ResNext models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnext50_tv --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnext50_tv_sgd_cy10by3_4cls

## Res2Net models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_48w_2s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_48w_2s_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_14w_8s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_14w_8s_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_26w_6s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_26w_6s_sgd_cy10by3_4cls

## Error:python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_26w_8s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_26w_8s_sgd_cy10by3_4cls

## RepVgg models
#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_A0 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_A0_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_A1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_A1_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_a2 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_a2_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b1_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b1g4 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b1g4_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b2 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b2_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b2g4 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b2g4_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b3 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b3_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save True --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name repvgg_b3g4 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_repvgg_b3g4_sgd_cy10by3_4cls

## So far BEST ResNext models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnext50_tv --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnext50_tv_sgd_cy10by3_4cls

## So far BEST MobileNet models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name mobilenetV2 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_mobilenetV2_sgd_cy10by3_4cls

## So far BEST ResNet models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnet18 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnet18_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name resnet34 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_resnet34_sgd_cy10by3_4cls

## So far BEST Res2Net models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_14w_8s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_14w_8s_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name res2net50_26w_6s --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_res2net50_26w_6s_sgd_cy10by3_4cls

## So far BEST Bit models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_bit_resnext50_1_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1_MOD --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_bit_resnext50_1_MOD_sgd_cy10by3_4cls

#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext101_1 --n_classes 4 --oversample 1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_bit_resnext101_1_sgd_cy10by3_4cls

## 5 classes TOR Training

## Bit models
#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext50_1 --n_classes 5 --oversample 1/1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_bit_resnext50_1_sgd_cy10by3_5cls

#python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext101_1 --n_classes 5 --oversample 1/1/1/1/1 --batch_size 8 --cycle_lens 10/3 --optimizer sgd --save_path TOR_bit_resnext101_1_sgd_cy10by3_5cls
## Checking for 15/3
python3 train_cyclical.py --do_not_save False --data_path /home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images --csv_train data/train_tor.csv --model_name bit_resnext101_1 --n_classes 5 --oversample 1/1/1/1/1 --batch_size 8 --cycle_lens 15/3 --optimizer sgd --save_path TOR_bit_resnext101_1_sgd_cy15by3_5cls
