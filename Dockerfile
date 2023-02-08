FROM python:3.9-slim
COPY model_deployment.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./pipeline_projet7.joblib /deploy/
COPY ./explainer.dill /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "model_deployment.py"]
