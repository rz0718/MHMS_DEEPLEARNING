# This Repo is about code and data for paper: ```Deep Learning and Its Applications to Machine Health Monitoring: A Survey```

## Table of Contents

<!-- TOC START min:1 max:3 link:true update:true -->
- [Personalized Recommendation](#personalized-recommendation)
  - [Table of Contents](#table-of-contents)
  - [Coocurrence Model](#coocurrence-model)
    - [Model training](#model-training)
  - [Recently View](#recently-view)
    - [General Logic](#general-logic)
    - [Steps (to update)](#steps-to-update)
  - [Demo of Snapshot data for Shopping_Cart](#demo-of-snapshot-data-for-shopping_cart)
  - [Production Data](#production-data)
  - [Update Schedule](#update-schedule)

<!-- TOC END -->



## Coocurrence Model

This model includes the coocurrence model for shopping cart recommendation. Four models are included in this project:


- Offline buy & order model over one year
- Offline click model over 1 month
- Online buy & order model (daily update)
- Online click model (daily update)

The offline & online models from the same action maintain the same underlying coocurrence matrix.

### Model training

#### For offline model:

```
cd coocurrence
spark-submit train/offline/tain.py --d <data_directory> --s <test_models> --a buy_order --l <test.log>
```

The ```<data_directory>```  should include a set of folders with name `Wtrain_matrix_order_shopeeBI*` or `Wtrain_matrix_buy_shopeeBI*`. The training script will look for data within such folders. An example of data directory format:

```
|__<data_directory>
|  |__Wtrain_matrix_order_shopeeBI_2017-01-01
|     |__part_0001.csv
|     |__part_0002.csv
|  |__Wtrain_matrix_buy_shopeeBI_2017_01_01
|     |__part_0001.csv
|     |__part_0002.csv

```
The csv files needs to contain header `userid` and `item`.

After training, the model will be saved in the following format.

```
test_models
|__coo_matrix.npz: co-occurrence matrix in scipy.sparse.csr_matrix format.
|__item2idx.pkl: pickled file of dictionary ({item: index})
|__idx2item.pkl: pickled file of dictionary({index: item})
|__part_*__coo_matrix.npz: intermediate results for co-occurrence matrix.
```

#### For online data (TO RUN DAILY UPDATES)

To run the airflow scheduler:
```
pip install airflow[all]
airflow initdb
airflow webserver -p <port>
python ~/airflow/dags/pipeline.py
airflow scheduler
```

Run the process from the UI.

- Daily update logic

![Daily Updates](images/coo_daily_updates.png)

- Weekly update logic

![Weekly Updates](images/coo_weekly_updates.png)



*** About daily updates ***

The click, buy & order data will be pulled to the following two folders daily.

```
/user/shopeeds/xiangyu/Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/Wtrain_matrix_buy_order_shopeeBI_golive

/user/shopeeds/xiangyu/Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/Wtrain_matrix_click_shopeeBI_golive
```

The logs will be saved to the following file.

```
/data/rx_exp/coocurrence_production_model/intermediate_outputs/logs/daily_update.log
```

Updated recommendations (intermediate outputs) will be saved to the following two files

```
/data/rx_exp/coocurrence_production_model/intermediate_outputs/daily_rec_click.txt

/data/rx_exp/coocurrence_production_model/intermediate_outputs/daily_rec_bo.txt
```

The daily backup will be saved to the following HDFS folder

```
/user/shopeeds/xiangyu/Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/backup/model_$DATE
```

The pipeline `pipeline.py` includes these following scripts:

- pull one day data

```
spark-submit $SCRIPT_HOME/online/scripts/pull_oneday_SG.py \
--s Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/ \
--l $SCRIPT_HOME/logs/daily_update.log
```

- train buy order data

```
spark-submit $SCRIPT_HOME/online/train_online.py \
--data-dir Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/Wtrain_matrix_buy_order_shopeeBI_golive \
--reference-model coocurrence_production_model/sg_1yr_daily_model \
--item2shop /data/rz_exp/shoppingcart_algo/db_pullcat/l2map_SG.csv \
--s daily_rec_bo.txt \
--info buy_order \
--l $SCRIPT_HOME/logs/daily_update.log
```

- train click data

```
spark-submit $SCRIPT_HOME/online/train_online.py \
--data-dir Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/Wtrain_matrix_click_shopeeBI_golive \
--reference-model coocurrence_production_model/sg_1yr_daily_model \
--item2shop /data/rz_exp/shoppingcart_algo/db_pullcat/l2map_SG.csv \
--s daily_rec_click.txt \
--info click \
--l $SCRIPT_HOME/logs/daily_update.log
```

- Write data to REDIS for buy & order model

```
python $SCRIPT_HOME/online/scripts/update_redis.py --n-jobs 50 \
--d daily_rec_bo.txt \
--l $SCRIPT_HOME/logs/daily_update.log \
--click 0
```

- Write data to REDIS for click model

```
python $SCRIPT_HOME/online/scripts/update_redis.py --n-jobs 50 \
--d daily_rec_click.txt \
--l $SCRIPT_HOME/logs/daily_update.log \
--click 1
```

## Recently View
This queue serves as part of the shopping cart recommendation.
### General Logic
- Based on users' behavior data in past 1 month + behavior data for today.
- Ranking:
    - score = x * popularity_score + y * relevance_score
    - Popularity score: Represent the popularity of the item itself.
    - Relevance score: Represent the relevance between the item and the user.
    - Score range = [0.1, 1.1]
- For now, items already ordered by the user in past 1 month are simply removed.
- On queue server:
    - Remove items already exist in shopping cart.
    - Adjust scores for items from the same shop as the recently added 5 shopping cart items. (x10)
    - Remove duplicates with the other 2 queues.
    - Merge with the other 2 queues and re-rank.
- On recommendation server:
    - Serveral filters applied, such as removing out-of-stock items and etc.

### Steps (to update)
- Initiate by preparing 1 month viewlog data, then run the following daily.

```
spark-submit --num-executors 20 --executor-cores 2 inhouseData_rv.py SG | tee logInhouseData.txt
```
- Continuous kafka data streaming from behavior topic.

```
spark-submit --queue streaming --num-executors 50 --executor-cores 4 --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 main_BehaviorKafkaToHDFS.py | tee logBehaviorKafkaToHDFS.txt
```
- Generate result and write to redis every 2-hour.

```
spark-submit --num-executors 50 --executor-cores 2 main_RecentlyView_v1.1.py SG | tee logRecentlyView.txt
```

## Demo of Snapshot data for Shopping_Cart

A demo of snapshot data is included in `demo_snapshot/`.

- `demo_snapshot/demo_data_merged_model.ipynb` is the file for generating demo data.
- Put the snapshot jsons into directory `demo_snapshot/snapshot_data`, and run ```python demo_snapshot/demo_snapshot.py```

## Production Data

| REGION | Data description | Model | Data path | Update frequency | Person in charge |
| --     | --               | --    | --        | --               | --               |
| SG     | Coocurrence production model | COOCURRENCE | daily | /data/rx_exp/coocurrence_production_model | Ruoxu |
| SG     | Item to shop information | COOCURRENCE & DSCF | weekly | /data/rz_exp/shoppingcart_algo/db_pullcat/l2map_SG.csv | Zhao Rui |
| TW     | Item to shop information | COOCURRENCE & DSCF | weekly | /data/rz_exp/shoppingcart_algo/db_pullcat/l2map_TW.csv | Zhao Rui |
| SG     | Daily backup of model | COOCURRENCE | daily | /user/shopeeds/xiangyu/Personal_Recommendation/data/goLive/shoppingcart/coocurrence/sg/backup/model_$DATE | Ruoxu |
| SG     | Behavior kafka streaming | DSRV | per min | /user/shopeeds/recommendation/behavior-kafka | Zixuan|
| SG     | Viewlog data | DSRV | daily | /user/shopeeds/recommendation/behavior-kafka | Zixuan|

## Update Schedule

- SG

DSRV: recent view
DSCO_w: coocurrence weekly updates
DSCO_d: coocurrence daily updates
CAT: category information updates

| Days / Hours | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|--------------|----------------------|---|-------------|--------|------|---|------|---|------|---|------|----|------|----|------|----|------|----|------|----|------|----|------|----|
| Mon |  |  | DSRV | DSCO_d | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Tue |  |  | DSRV | DSCO_d | DSRV | DSCF | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Wed |  |  | DSRV DSCO_w | DSCO_d | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Thu | 00:30 CAT 00:00 DSRV |  | DSRV | DSCO_d | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Fri | DSRV |  | DSRV | DSCO_d | DSRV | DSCF | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Sat | DSRV |  | DSRV | DSCO_d | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |
| Sun | DSRV |  | DSRV | DSCO_d | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  | DSRV |  |