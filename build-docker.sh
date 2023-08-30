CVLCORE_IMAGE_TAG=opencv_ubuntu
CVLCORE_CONTAINER_NAME=$CVLCORE_IMAGE_TAG-container

echo -e "CVLCore image tag is: $CVLCORE_IMAGE_TAG"
echo -e "CVLCore container name is: $CVLCORE_CONTAINER_NAME"

docker build --force-rm -f Dockerfile -t $CVLCORE_IMAGE_TAG .
docker run -d --name $CVLCORE_CONTAINER_NAME $CVLCORE_IMAGE_TAG
docker cp $CVLCORE_CONTAINER_NAME:/home/user/cvlcore/cvlcore.tar.gz ./target/docker/

