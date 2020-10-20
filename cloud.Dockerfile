# adaptcloud/Dockerfile
FROM adapt

# to build:
#   docker build . -t adapt
#   docker build . -t adaptcloud -f ./cloud.Dockerfile
#
# to run with memo:
#   docker run --rm  -v /path/to/memo/on/host:/memo <image_ID> "design.py subcommands"
#
# to run interactively:
#   docker run --rm -it <image_ID>
#

COPY ./requirements-with-aws.txt .
RUN pip install -r requirements-with-aws.txt

CMD "/bin/bash"
