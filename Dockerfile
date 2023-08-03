FROM python

COPY final.py .
COPY requirements.txt .
COPY dataset.csv .

RUN pip install -r requirements.txt

CMD ["python", "final.py"]