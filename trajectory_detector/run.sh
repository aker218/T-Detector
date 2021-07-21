# Install the required package
pip install -r requirements.txt

echo 'preprocessing...'
# Convert the original data to the correct format
python -u preprocess.py  -D ./data/new_processed_data/ -M ./data/new_dataset/mouse/ -ML ./data/new_dataset/ -L ./data/new_dataset/move/ -LL ./data/new_dataset/ -P0 0 -P1 0

echo 'time_dis_w2v_preprocess...'
# LocationTime2vec preprocess
python -u time_dis_w2v_preprocess.py -D ./data/new_processed_data/ -M

echo 'time_dis_w2v...'
# game character trajectory's LocationTime2vec pretrain
python -u time_dis_w2v.py -D ./data/new_processed_data/ -T location -S -DV 0 -M
# mouse trajectory's LocationTime2vec pretrain
python -u time_dis_w2v.py -D ./data/new_processed_data/ -T mouse -S -DV 0 -M

echo 'w2v...'
# normal word2vec pretrain of two kinds of trajectory
python -u w2v.py -D ./data/new_processed_data/ -M

echo 'making dataset...'
# LocationTime2vec based dataset building
python -u make_dataset.py -D ./data/new_processed_data/ -M -T
# normal word2vec based dataset building
python -u make_dataset.py -D ./data/new_processed_data/ -M

# no pretrained embedding based T-detector’s Angle Pretrain
echo 'angle_pretrain...'
python -u angle_pretrain.py -D ./data/new_processed_data/ -M -DV 0
# Word2vec pretrained embedding based T-detector’s Angle Pretrain
python -u angle_pretrain.py -D ./data/new_processed_data/ -M -DV 0 -P
# WLocationTime2vec pretrained embedding based T-detector’s Angle Pretrain
python -u angle_pretrain.py -D ./data/new_processed_data/ -M -DV 0 -P -T

echo 'runing models...'
# model comparison
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP -P
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP -P -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP -P -T
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT MLP -P -T -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN -P
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN -P -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN -P -T
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT CNN -P -T -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU -P
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU -P -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU -P -T
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT BiGRU -P -T -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -P
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -P -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -P -T
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -P -T -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP -P
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP -P -TL
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP -P -T
python -u train_and_evaluate.py -D ./data/new_processed_data/ -M -DV 0 -MT ConvGRU -A -MAT -AP -P -T -TL

