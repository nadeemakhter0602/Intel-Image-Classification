# Intel Image Classification

Training a model on [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle. Focus is on making a small and lightweight model which is also reasonably accurate on unseen test data.

Accuracy on test data after 20 epochs was 0.9170, with a model file size of around 19 megabytes (mobilenet_128.h5).

Training and Validation Accuracy Graph :

![](assets/Figure_1.png)

Training and Validation Loss Graph :

![](assets/Figure_2.png)

Entire Model Architecture as exported by [Netron Viewer](https://github.com/lutzroeder/netron) :

![](assets/mobilenet_128.png)
