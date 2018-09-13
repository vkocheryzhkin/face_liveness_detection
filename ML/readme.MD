# Prerequisites:

### Download 

shape_predictor_68_face_landmarks.dat from http://dlib.net/files

### Train:

1. Convert at least two videos into frames (move camera against face and printed face)

```shell
	python preprocess_movies.py -i ~/data/live.MOV -o ~/data/live 
	python preprocess_movies.py -i ~/data/fraud.mp4 -o ~/data/fraud
```

2. Generate flow using all pairs

```shell
	python get_optical_flow.py -i ~/data/live -o ~/data/live_of
	python get_optical_flow.py -i ~/data/fraud -o ~/data/fraud_of
```

3. Train 

```shell
	python train_keras.py --live "~/data/live_of" --fraud "~/data/fraud_of"
```

### Test:

```shell
python test_keras.py -l ~/data/live/0030.jpg -r ~/data/live/0010.jpg
```