FROM python:3.8

WORKDIR /home/app


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY . /home/app

EXPOSE 8081

ENV PORT=8081
ENV DEBUG=False

CMD [ "gunicorn", "wsgi:app"]