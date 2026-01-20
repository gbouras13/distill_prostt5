
#
# docker build --build-arg GITHUB_TOKEN={token}  -t distill_prostt5:0.2.0 .  
#

# FROM quay.io/pawsey/rocm-mpich-base:rocm6.3.3-mpich3.4.3-ubuntu24.04
# FROM rocm/pytorch:rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

# original
# FROM quay.io/pawsey/rocm-mpich-base:rocm6.3.1-mpich3.4.3-ubuntu22


# ARG LIBFABRIC_VERSION=1.18.1

# # Install required packages and dependencies
# RUN   apt -y update \
#       && apt -y install build-essential wget doxygen gnupg gnupg2 curl apt-transport-https software-properties-common  \
#  git vim gfortran libtool python3-venv ninja-build \
#       libnuma-dev python3-dev \
#       && apt -y remove --purge --auto-remove cmake \
#       && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null\
#  | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
#       && apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ jammy-rc main" \
#       && apt -y update 

# # Build and install libfabric
# RUN (if [ -e /tmp/build ]; then rm -rf /tmp/build; fi;) \
#       && mkdir -p /tmp/build \
#       && cd /tmp/build \
#       && wget https://github.com/ofiwg/libfabric/archive/refs/tags/v${LIBFABRIC_VERSION}.tar.gz \
#       && tar xf v${LIBFABRIC_VERSION}.tar.gz \
#       && cd libfabric-${LIBFABRIC_VERSION} \ 
#       && ./autogen.sh \
#       && ./configure \
#       && make -j 16 \ 
#       && make install

# #
# # Install miniforge
# #

# RUN set -eux ; \
#   curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh ; \
#   bash ./Miniforge3-* -b -p /opt/miniforge3 -s ; \
#   rm -rf ./Miniforge3-*
# ENV PATH /opt/miniforge3/bin:$PATH
# RUN conda clean -af -y



# FROM quay.io/gbouras13/distill_prostt5:0.4.1

FROM quay.io/pawsey/pytorch:2.7.1-rocm6.3.3

ARG GITHUB_TOKEN

RUN echo test13
# uninstall the v0.4.1 install
# RUN pip uninstall -y distill_prostt5 || true
# RUN echo installing 2

RUN pip install git+https://${GITHUB_TOKEN}@github.com/gbouras13/distill_prostt5.git@main
RUN python3 -c "import torch; print(torch.__version__)" || echo "PyTorch is not installed before installing the project. "
RUN distill_prostt5 -h
RUN distill_prostt5 train -h
RUN distill_prostt5 infer -h

