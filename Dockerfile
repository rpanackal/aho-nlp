FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

LABEL maintainer="rr186@mars.uni-freiburg.de"

# set a directory for the app
WORKDIR /ahpo

# copy all the files to the container
COPY . ./
RUN chmod -R a+rwx .

# install dependencies
RUN apt-get -y update
RUN apt-get -y install apt-utils python3-pip make vim bash-completion
RUN pip3 install --no-cache-dir torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir torchtext
RUN pip3 install --no-cache-dir -U scikit-learn
RUN pip3 install --no-cache-dir 'ray[tune]'
RUN pip3 install --no-cache-dir -U pip setuptools wheel
RUN pip3 install --no-cache-dir -U spacy[cuda110]
RUN python3 -m spacy download en_core_web_sm
RUN pip3 install --no-cache-dir hpbandster ConfigSpace

EXPOSE 8265


#To build a docker image
#docker build -t roshin-panackal-demo .

#To create a container with GPU access
#mount directory /local/data/rajanro/embeddings in machine rubur or /nfs/students/deepnlp2021/adsem13/embeddings
#docker run -it -v /nfs/students/deepnlp2021/adsem13/embeddings:/ahpo/embeddings --name roshin-panackal-demo  roshin-panackal-demo
