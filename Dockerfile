FROM python:3.8.5-slim

LABEL maintainer='Priya Pillai <ppillai@broadinstitute.org>'

# to build:
#   docker build . -t adapt
#
# to run with memo:
#   docker run --rm  -v /path/to/memo/on/host:/memo <image_ID> "design.py subcommands"
#
# to run interactively:
#   docker run --rm -it <image_ID>
#

ENV \
    WORK_DIR=/adapt \
    MEMO_DIR=/memo \
    MAFFT_PATH=/usr/bin/mafft

ENV OUTPUT_DIR=$WORK_DIR/output

WORKDIR $WORK_DIR

RUN mkdir $MEMO_DIR
RUN mkdir $OUTPUT_DIR

RUN apt update \
    && apt-get install -y wget

RUN wget https://mafft.cbrc.jp/alignment/software/mafft_7.471-1_amd64.deb \
    && dpkg -i mafft_7.471-1_amd64.deb \
    && rm -rf mafft_7.471-1_amd64.deb

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD "/bin/bash"
