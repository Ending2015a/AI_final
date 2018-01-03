# End to End Memory Net for CBTests
## Requirements
* Python 3
* tensorflow
* tqdm
Install by pip
```
pip install -r requirements.txt
```

## Demo
### Dataset
1. First you need to get the CBTest dataset from [here (AI_Course_Final.zip)](https://www.kaggle.com/c/itic2017practice/data)

2. Extract
```
unzip AI_Course_Final.zip
```
3. Run preprocess.py to generate dataset, this may take a while.
```
python3 preprocess.py -p all
```
4. Configure network settings by modifying train.py

5. Start training
```
python3 train.py
```
