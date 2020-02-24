
train:
	./train_local.sh text-generation:latest
.PHONY: train

train2:
	./train_local.sh text-generation-2units-2layers:latest 
.PHONY: train2

serve:
	./serve_local.sh text-generation:latest
.PHONY: serve

predict:
	#cd SageMaker/amazon-sagemaker-keras-text-classification/container/local_test 
	#cat input.json 
	./predict.sh input.json application/json
.PHONY: predict
