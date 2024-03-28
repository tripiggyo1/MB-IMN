# MB-IMN
Public implementation of the Multi-behavior Interest Matching Network (MB-IMN), ICWS 2024


# Requirements
The code is built on Pytorch and the [Recbole](https://github.com/RUCAIBox/RecBole) benchmark library. Run the following code to satisfy the requeiremnts by pip:

    pip install -r requirements.txt

# Datasets
- The dataset folder contains a processed and ready-to-use dataset, you can also process other datasets into same format and move them to ```./dataset/```.

- The datasets used in this paper are available at:

  [UserBehavior](https://tianchi.aliyun.com/dataset/649)

  [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

  [IJCAI](https://tianchi.aliyun.com/dataset/42)

# Run MB-IMN

```python run_MB-IMN.py --model=[MBIMN] --dataset=[retail_beh] --gpu_id=[0] --batch_size=[2048]```, where [value] means the default value.

# Tips
- Note that we modified the evaluation sampling setting in ```recbole/sampler/sampler.py``` to make it static.
- The model code is at ```recbole/model/sequential_recommender/mbimn.py```.
- The hyperparameter setting is at ```recbole/properties/model/MBIMN.yaml```.
- Feel free to explore other baseline models provided by the RecBole library and directly run them to compare the performances.




