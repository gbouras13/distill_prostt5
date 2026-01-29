
# FROM quay.io/gbouras13/distill_prostt5:0.4.1
# note that previously, v0.4.1 used the pytorch nightly - easier just to build on top of the validated pawsey build

FROM quay.io/pawsey/pytorch:2.7.1-rocm6.3.3

ARG GITHUB_TOKEN

RUN echo test13
# uninstall the v0.4.1 install
# RUN pip uninstall -y distill_prostt5 || true
RUN echo installing 32

RUN pip install git+https://${GITHUB_TOKEN}@github.com/gbouras13/distill_prostt5.git@main
RUN python3 -c "import torch; print(torch.__version__)" || echo "PyTorch is not installed before installing the project. "
RUN distill_prostt5 -h
RUN distill_prostt5 train -h
RUN distill_prostt5 infer -h

