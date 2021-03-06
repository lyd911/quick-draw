## Quick, Draw！

Using CNN model to do sketch recognition, based on Google quick draw dataset https://quickdraw.withgoogle.com/data, codes modified from https://github.com/akshaybahadur21/QuickDraw. This repo doesn't include dataset


### Dataset
You can either download from **BaiduNetDisk**(https://pan.baidu.com/s/1PAkFyH8EV-JES2zhB_5xBw) or from **Google Cloud Platform**(https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap)

I got `.npy` files from google cloud for 10 drawings ['candle', 'door', 'lightning', 'moon', 'mountain', 'shoes', 'sword', 't-shirt', 'telephone', 'train']


### Procedure

1) Get the dataset as mentioned above and place the `.npy` files in `/data` folder.
2) First, run `QD_trainer.py` which will load the data from the `/data` folder and store the features and labels in  pickel files. Then load data from pickle and augment it. After this, the training process begins. The output is trained model file 'QuickDraw.h5'.
3) Now you need to test the model, run `QuickDrawApp.py` which will take the test file stored in source folder as the input, and get the prediction.


### Requirements

python: 3.6.5  
pip: 19.3.1  
tensorflow: 2.0.0  
numpy: 1.16.1  
scikit-learn: 0.22  
keras: 2.3.1  
opencv-python: 4.1.2  

### References:
 
 - [Google's Quick, Draw](https://quickdraw.withgoogle.com/) 
 - [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
 - [Quick Draw: the world’s largest doodle dataset](https://towardsdatascience.com/quick-draw-the-worlds-largest-doodle-dataset-823c22ffce6b)
