FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG HOME='/home/ds/'
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make \
    build-essential \ 
    libssl1.1 \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils tk-dev \
    libffi-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    texlive-xetex \ 
    libgl1-mesa-glx \
    r-base		

WORKDIR ${HOME}

# copy file with packages requeriments
COPY requeriments.txt ${HOME}requeriments.txt

# install pyenv and python
#RUN git clone git://github.com/yyuu/pyenv.git .pyenv

#ENV PYENV_ROOT ${HOME}/.pyenv
#ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

#RUN pyenv install 3.8.1
#RUN pyenv global 3.8.1
#RUN pyenv rehash

# upgrade pip
RUN pip  install --upgrade pip 

## install python packages
RUN pip install -r requeriments.txt

# remove requeriments.txt
RUN rm requeriments.txt

CMD ["python3"]
