# initial set up (the three lines we always have to start with)
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# import the fastai vision library
from fastai.vision import *
from pathlib import Path

# create the appropriate path variable that points to the fer2013 directory in
# your data directory
path = Path('data/expressions/fer2013')
path

# import pandas and numpy
import pandas as pd 
import numpy as np

# create a pandas object from our csv
df_label = pd.read_csv(path/'fer2013.csv')

# loop through our entire dataset, change redundant emotion labels to their
# counterpart
j = 0
while j < 35887:
    if df_label.loc[j, 'emotion'] == 5:
        df_label.loc[j, 'emotion'] = 2
    if df_label.loc[j, 'emotion'] == 6:
        df_label.loc[j, 'emotion'] = 4
    if df_label.loc[j, 'emotion'] == 1:
        df_label.loc[j, 'emotion'] = 0
    j += 1

# drop pixel label
df_label.drop(['pixels'],axis=1,inplace=True)

# drop all of the testing pictures in our entire dataset so we can train
# the model with our training pictures
i = 28710
while i<35887:
    df_label.drop([i], inplace=True)
    i += 1

# modify our dataset to include the labels we need for training. We want 
# one column to have the title "Image_.jpg" followed by its corresponding
# emotion in the adjacent column
df_label.drop([28709], inplace=True)
df_label.drop(['Usage'],axis=1,inplace=True)
df_label.insert(0,'image', range(1,28710))
df_label['image'] = df_label['image'].astype(str)+'.jpg'


# convert our dataset back to a csv file
df_label.to_csv('data/expressions/fer2013/train.csv',index=False)

# create our data bunch object from the csv using a folder of training
# photos to represent our image column
np.random.seed(42)
data = (ImageItemList.from_csv(path, 'train.csv', folder='Training')
       .random_split_by_pct(0.2)
       .label_from_df(cols='emotion')
       .databunch().normalize(imagenet_stats))

# create our model using ResNet 50
learn = create_cnn(data, models.resnet50, metrics=error_rate)

# commands to find an ideal learning rate
learn.lr_find()
learn.recorder.plot()
lr = 0.5e-01

# fit our model using the calculated learning rate and 5 epochs
learn.fit_one_cycle(5, slice(lr))

# save as stage 1, unfreeze so we can further train it
learn.save('stage-1-rn50')
learn.unfreeze()

# find new learning rate for trained model, fit our model again
# using updated learning rate of 1e-5
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5))

#save this model as stage-2 and export
learn.save('stage-2-rn50')
learn.export()
