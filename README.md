# Image-and-Spatial Transformer Networks
This is an implementation of the method described in paper

```
@inproceedings{lee2010istn,
    author = {Lee, Matthew C.H. and Oktaz, Ozan and Schuh, Andreas and Schaap, Michiel and Glocker, Ben},
    title = {Image-and-Spatial Transformer Networks for Structure-guided Image Registration},
    year = {2019},
    book = {International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)}
}
```

All rights reserved. Copyright 2019

To unzip the 2D synthetic data, from the parent directory run

`unzip data/synth2d/unzip_here.zip -d data/synth2d/`

To install requirements run

`pip install -r requirements.txt`

To run a 2D experirment with and explicit ISTN on the synthetic data run

`python istn-reg.py --config data/synth2d/config.json --transformation affine --loss e --out output/stn-e --model output/stn-e/train/model`

`--loss` controls the type of loss used, options are

`u` - unsupervised
`s` - supervised
`e` - explicit
`i` - implicit

Tensorboard logs are placed in `{{--out}}/tensorboard`, to fire up tensorboard you can run

`tensorboard --logdir {{--out}}`

If you installed tensoboard using pip, you may not be able to call it as above as. You can find it's installation location by running

`pip show tensoboard`

Then you can run 

`python tensorboard/location/main.py --logdir {{--out}}`

Models are saved under `{{--out}}/train`, once training is complete test results are found processed and put into `{{--out}}/test`
