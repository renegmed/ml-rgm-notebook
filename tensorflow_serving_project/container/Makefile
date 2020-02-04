build:
	docker build -t sagemaker-tf-serving .
.PHONY: build

run:
	docker run --rm -p 8080:8080 sagemaker-tf-serving
.PHONY: run

call:
	curl -X POST http://localhost:8080/invocations -d '{"instances": [1.0,2.0,5.0]}'
.PHONY: call

