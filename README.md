# Style transfer
This is tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'.

## Download program
```
$ git clone https://github.com/fullfanta/real_time_style_transfer.git
```

## Download training data
```
$ cd real_time_style_transfer
$ sh get_coco.sh
```

## Download vgg16 model
```
$ sh get_vgg16.sh
```

## Train
```
$ python train.py
```
If you have multiple GPU cards, use CUDA_VISIBLE_DEVICES to specify GPU card.
Trained model is in summary.
To adjust weights of content loss and style loss, you can set the parameter alpha. loss equation is alpha * content loss + (1 - alpha) * style loss. Default is 0.1
During training, you can see generated images through tensorboard.
```
$ tensorboard --logdir=summary
```


## Freeze model
```
$ sh freeze.sh
```
It generates pb file which contains weights as contant.

## Test
```
$ python stylizer.py --model=models/starry_night.pb --input_image=test_images/Aaron_Eckhart_0001.jpg
```
It generates stylized image and save it to 'test_images/Aaron_Eckart_0001_output.jpg


## Examples
|    | Input | Output |
|----|-------|--------|
|Aaron Echart|<img src='test_images/Aaron_Eckhart_0001.jpg' width='256px'>|<img src='test_images/Aaron_Eckhart_0001_output.jpg' width='256px'>|
|Angelina Jolie|<img src='test_images/jolie.jpg' width='256px'>|<img src='test_images/jolie_output.jpg' width='256px'>|
|Dinosour|<img src='test_images/dinosour.png' width='256px'>|<img src='test_images/dinosour_output.jpg' width='256px'>|
|Ryan|<img src='test_images/ryan.png' width='256px'>|<img src='test_images/ryan_output.jpg' width='256px'>|
|Herb|<img src='test_images/herb.png' width='256px'>|<img src='test_images/herb_output.jpg' width='256px'>|
|Cheez|<img src='test_images/cheez.png' width='256px'>|<img src='test_images/cheez_output.jpg' width='256px'>|
# multimodal_transfer
