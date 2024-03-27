# MB-IMN
Public implementation of the Multi-behavior Interest Matching Network (MB-IMN), ICWS 2024


# Requirements
The code is built on Pytorch and the [Recbole](https://github.com/RUCAIBox/RecBole) benchmark library. Run the following code to satisfy the requeiremnts by pip:

    pip install -r requirements.txt

# Downloading datasets
The datasets used in this paper are available at:

[UserBehavior](https://tianchi.aliyun.com/dataset/649)

[Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

[IJCAI](https://tianchi.aliyun.com/dataset/42)

# Run MB-IMN

    python run_MBHT.py --model=[MBHT] --dataset=[retail_beh] --gpu_id=[0] --batch_size=[2048], 
where [value] means the default value.


