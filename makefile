# im2katex model
im2katex-trainval-aug:
	python3 main.py --mode trainval --data_type merged  --model_type im2katex  --gpu=0  --encoder_type Augment
	
im2katex-trainval-ori:
	python3 main.py --mode trainval --data_type merged  --model_type im2katex  --gpu=0  

im2katex-eval:
	python3 main.py --mode val  --data_type merged --model_type im2katex --gpu=0

im2katex-test:
	python3 main.py --mode test  --data_type merged --model_type im2katex  --gpu=0

im2katex-inference:
	python3 main.py --mode infer --predict_img_path  /home/xiaofeng/data/image2latex/handwritten/process/img_padding --data_type merged --model_type im2katex --gpu=0

# erroer model
error-trainval:
	python3 main.py --mode trainval  --model_type error

error-eval:
	python3 main.py --mode val  --model_type error

error-test:
	python3 main.py --mode test  --model_type error

error-inference:
	python3 main.py --mode infer --predict_img_path  /home/xiaofeng/data/image2latex/handwritten/process/img_padding --model_type im2katex


server:
	python3 app.py --data_type merged --gpu=0

# full: build train eval
