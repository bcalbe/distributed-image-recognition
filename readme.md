
## Split VGG19 into 10 submodel
Basically the method is very simple. I reconstruct a dataset which consist of the target class images as positive samples and I evenly pick up images from the other class as the negative samples. In the reconstructed
dataset, the amount of the positive samples and negative samples are the same.

## Version 1
So far, I have successfully retrain the 10 submodels and the accuracy are shown as below. At the same time I sequentially run each model with the same input. Once I obtain an output higher than the threshold(right now 0.7), we can early stop and output the results. 
However, if we cannot get a output higher than threshold after all models have been run, we output the largest results. Under this situation, we cannot infer the results through the model runniung

    the accuracy of model 0 is  0.9332
    the accuracy of model 1 is  0.9641
    the accuracy of model 2 is  0.92
    the accuracy of model 3 is  0.8313
    the accuracy of model 4 is  0.9033
    the accuracy of model 5 is  0.9363
    the accuracy of model 6 is  0.9643
    the accuracy of model 7 is  0.9267
    the accuracy of model 8 is  0.9739
    the accuracy of model 9 is  0.9628

### Problems so far
The output of each model is not reliable enough. I try to input the image with classes from 0-9 and the results are as follows

    time cost for class 0,0 is 3.384890079498291
    time cost for class 1,1 is 0.26284241676330566
    time cost for class 2,2 is 0.316300630569458
    time cost for class 3,3 is 0.32062315940856934
    time cost for class 4,0 is 0.05549359321594238
    time cost for class 5,5 is 0.5777533054351807
    time cost for class 6,4 is 0.20015239715576172
    time cost for class 7,7 is 0.8916459083557129
    time cost for class 8,1 is 0.08362817764282227
    time cost for class 9,1 is 0.07866168022155762

The left num refers to the ground truth and the right number is the predicted class. From the correct results 1,2,3,5,7, we can see that the inference time is increased as the classes increased.

