FROM ubuntu

RUN apt-get update -o Acquire::Check-Valid-Until=false
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3 python3-pip

EXPOSE 3001

RUN mkdir -p /root/.local/share/pwr/
COPY config.yaml /config.yaml
