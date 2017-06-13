# SimGAN
Keras implementation of Apple's [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf)

## Running

Install dlutils from https://github.com/wayaai/deep-learning-utils:

> $ pip install -U git+https://github.com/wayaai/deep-learning-utils.git  

or  

> $ git clone https://github.com/wayaai/deep-learning-utils.git  
> $ python setup.py install develop


```python
python3 sim-gan.py PATH_TO_SYNTHESEYES_DATASET PATH_TO_MPII_GAZE_DATASET
```

In apple's paper they use [Unity Eyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) to generate ~1.2 million synthetic images. I am on mac though so I just used the easily available [SynthesEyes Dataset](https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/). This is small (only around ~11,000 images) so it would be much better if someone could generate a larger dataset w/ [Unity Eyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) and share it on s3.

The dataset of real image's used in apple's paper is the [MPIIGaze Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/). They use the normalized images provided in this dataset which are stored in matlab files. It was a bit of a pain to get these in an easily usable form so I'm sharing the ready to go datasets on s3.

## Ready to go datasets
* 50000 Unity Eyes: https://www.kaggle.com/4quant/eye-gaze

## Details
Implementation of `3.1 Appearance-based Gaze Estimation` on UnityEyes and MPIIGaze datasets as described in paper.

* Currently only Python 3 support.
* Tensorflow support and maybe PyTorch support in future.

## Implementation
This is meant to be a light-weight and clean implementation that is easy to understand - no deep shit. It can also be used as a resource to understand GANs in general and how they can be implemented.

## Running Online

You can see a interactive Jupyter Notebook version of this script with training data on [Kaggle](https://www.kaggle.com/kmader/d/4quant/eye-gaze/simgan-notebook/) or just the [raw training set](https://www.kaggle.com/4quant/eye-gaze)

## About waya.ai
Waya.ai is a company whose vision is a world where medical conditions are addressed early on, in their infancy. This approach will shift the health-care industry from a constant fire-fight against symptoms to a preventative approach where root causes are addressed and fixed. Our first step to make realize this vision is easy, accurate and available diagnosis. Please get in contact with me if this resonates with you!
