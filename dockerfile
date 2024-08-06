FROM python:3.11

COPY /sub .

RUN pip install -r requirements.txt

CMD ["python3", "sub.py"]