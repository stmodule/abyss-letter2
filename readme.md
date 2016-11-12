# Setup

1. Put "ETL2" on root directory.
1. Run `extract.py`. npy files will be created in "ETL2_npy".

# DCGAN

1. Run `./dcgan/make_data.py` to create training data.
1. Run `./dcgan/train_dcgan.py` to train model. Periodically save model files in "./dcgan/models".
1. Run `./dcgan/predict_dcgan.py dcgan/models/XX.h5` to show randomly generated images.