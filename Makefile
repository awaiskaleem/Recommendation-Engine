CURRENT_DIR := $(shell pwd)

NAME = Recommendation Engine
N?=100

build:
	@docker-compose up --build

launch:
	@echo "Starting ${NAME}..."
	@docker-compose up
    
shutdown:
	@echo "Shutting Down ${NAME}..."
	@docker-compose down

help:
	@echo "Following are the \"make\" options available:"
	@echo "====>build -- build docker image for recommendation engine"
	@echo "====>train -- this will train using data available in ./data/ directory"
	@echo "====>predict user_id N -- will give back N predictions for user_id"
	@echo "====>predict_file - will add predictions to ./data/predictions.csv and save it to ./output/"
	@echo "====>print this for Klarna :)"
    
print:
	@echo "this"
    
train:
	@curl http://0.0.0.0:5000/train

predict:
	@curl http://0.0.0.0:5000/predict/${user_id}/${N}

predict_file:
	@curl http://0.0.0.0:5000/predict_file/${N}