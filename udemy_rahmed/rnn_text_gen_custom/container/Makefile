login:
	aws ecr get-login --no-include-email --region us-east-1 --registry-ids 763104351884
.PHONY: login

build:
	docker build -t text-generation:latest .
.PHONY: build

build2:
	docker build -t text-generation-2units-2layers:latest .
.PHONY: build2

copy_data:
	cp -a ../data local_test/test_dir/input/data/training/
.PHONY: copy_data
