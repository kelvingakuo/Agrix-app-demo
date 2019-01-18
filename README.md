## SETTIING UP THE ENVIRONMENT
1. Create a virtualenv e.g. using Conda, running Python 3.6
```bash
conda create -n nameofEnv python=3.6
```
2. Activate the env
```bash
source activate nameofEnv
```
3. Install requirements
```bash
pip install -r requirements.txt
```

4. Create needed dirs
```bash
mkdir data/crowdai_train
mkdir data/crowdai_test
mkdir saved_models
```

## DATA
1. Download train.tar and test.tar from crowdai.org/challenges/1 under 'Datasets'
2. Extract train.tar into 'data/crowdai_train' and test.tar into 'data/crowdai_test'


## USAGE
1. Run
```bash
python train_nn.py modelName
```

The modelName is either 'alexnet' or 'vgg16'

2. Wait
