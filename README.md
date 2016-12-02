# cs229final
The final project of CS229

### preparing data
download the data from kaggle to `data` directory and unzip
```
data/images/
data/sample_submission.csv
data/test.csv
data/train.csv
```

### software prerequiste
python 2 w/o gpu:
```bash
brew install opencv3 --HEAD --with-contrib
echo /usr/local/opt/opencv3/lib/python2.7/site-packages > $PYTHON_PATH/site-packages/opencv3.pth
pip install -U https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl
pip install -r requirements
```

python 3 w/ gpu:
```bash
brew install opencv3 --with-contrib --with-cuda --with-python3 --c++11
echo /usr/local/opt/opencv3/lib/python3.5/site-packages > $PYTHON_PATH/site-packages/opencv3.pth
pip install -U https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0-py3-none-any.whl
pip install -r requirements
```

### bootstrap
in command line enter the following:
```bash
jupyter notebook
```
a browser will open to show the jupyter notebook, select the `ipynb` files to open one of them.

### dataset

1. Kaggle
    [Kaggle Leaf Classification](https://www.kaggle.com/c/leaf-classification)

2. UCI
    [UCI Leaf Dataset](https://archive.ics.uci.edu/ml/datasets/Leaf)

3. Swedish leaf dataset
    [Swedish Leaf Dataset](http://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/)
    The files are named so that it is easy to find the leaf/leaves that we want. A file could be named for example :
    `l9nr003.tif`
    
    | symbol | explanation         |
    | ------ | :------------------ |
    | l      | is short for leaf   |
    | 9      | is treeclass nine   |
    | nr     | is short for number |
    | 003    | is the number of the leaf of the specific treeclass |

    Oskar J. O. Söderkvist, “Computer vision classifcation of leaves from swedish trees,” Master’s Thesis, Linkoping University, 2001.

    15 labels, >60 samples per label.

4. Flavia data set
    [Flavia Dataset](http://flavia.sourceforge.net)
    Please cite it as the data used in our paper: Stephen Gang Wu, Forrest Sheng Bao, Eric You Xu, Yu-Xuan Wang, Yi-Fan Chang and Chiao-Liang Shiang, A Leaf Recognition Algorithm for Plant classification Using Probabilistic Neural Network, IEEE 7th International Symposium on Signal Processing and Information Technology, Dec. 2007, Cario, Egypt

    33 labels. ~60  samples per label.

    Leafsnap dataset, Intelengine dataset, and ImageCLEF

# sample commandline
imageclef end to end instruction:
```bash
python src2/organizer.py -s data/imageclef/ImageCLEF2013PlantTaskTrainPackage-PART-1/train/ -d data/imageclef/train
cp -rv data/imageclef/TestAndTaskPackage/Data/GroundTruth data/imageclef/TestAndTaskPackage/Data/Test
python src2/organizer.py -s data/imageclef/TestAndTaskPackage/Data/Test -d data/imageclef/test
python src2/main.py extract -f history/imageclef_train_resnet50.h5 -a resnet50 conf/imageclef-train.yaml
python src2/main.py extract -f history/imageclef_test_resnet50.h5 -a resnet50 conf/imageclef-test.yaml
python src2/main.py classify -f history/imageclef_train_resnet50.h5 -t history/imageclef_test_resnet50.h5 -m history/imageclef_model.pkl
```

# reference

- [Wikipedia: Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [2014 about feature scaling](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
- [CS231N: Neural Network Data Preparation](http://cs231n.github.io/neural-networks-2/#datapre)
- [OpenCV 2d Feature Extraction](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)
- [From feature descriptors to deep learning: 20 years of computer vision](http://www.computervisionblog.com/2015/01/from-feature-descriptors-to-deep.html)


# appendix: opencv caveats:
This formula is keg-only, which means it was not symlinked into /usr/local.

opencv3 and opencv install many of the same files.

Generally there are no consequences of this for you. If you build your
own software and it requires this formula, you'll need to add to your
build variables:

    LDFLAGS:  -L/usr/local/opt/opencv3/lib
    CPPFLAGS: -I/usr/local/opt/opencv3/include
    PKG_CONFIG_PATH: /usr/local/opt/opencv3/lib/pkgconfig

If you need Python to find bindings for this keg-only formula, run:

    echo /usr/local/opt/opencv3/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth
    mkdir -p
    echo 'import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")' >> homebrew.pth
