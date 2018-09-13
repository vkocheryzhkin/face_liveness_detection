Prerequisites:

1. Download shape_predictor_68_face_landmarks.dat

Train:

1. Convert least two videos into frames (move camera against face or printed face)
	python preprocess_movies.py -i ~/data/live.MOV -o ~/data/live 
	python preprocess_movies.py -i ~/data/fraud.mp4 -o ~/data/fraud

2. Get flow using all pairs
	python get_optical_flow.py -i ~/data/live -o ~/data/live_of
	python get_optical_flow.py -i ~/data/fraud -o ~/data/fraud_of

3. Train 
	python train_keras.py --live "~/data/live_of" --fraud "~/data/fraud_of"

Test:

python test_keras.py -l ~/data/live/0030.jpg -r ~/data/live/0010.jpg