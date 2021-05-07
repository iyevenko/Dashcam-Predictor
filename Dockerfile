#FROM rocm/tensorflow:latest
FROM rocm/tensorflow:rocm3.10-tf2.3-dev

RUN apt install rocm-libs rccl
RUN pip3 install tensorflow-rocm==2.3.1
RUN pip3 install --user matplotlib
RUN pip3 install --user opencv-python

RUN mkdir -p /root/data
RUN mkdir -p /root/saved_models
RUN mkdir -p /root/plots
RUN mkdir -p /root/logs
COPY ./dashcam_predictor dashcam_predictor/
COPY ./saved_models saved_models/
COPY train.py setup.py predict.py test.py ./
RUN python3 setup.py build
RUN python3 setup.py install --user

ENTRYPOINT python3 ./train.py