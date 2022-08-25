IMAGE_NAME=tybalex/opni-rca:mc
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME

IMAGE_NAME2=tybalex/opni-rca_trainer:mc
docker build . -t $IMAGE_NAME2 -f ./Dockerfile_trainer

docker push $IMAGE_NAME2
