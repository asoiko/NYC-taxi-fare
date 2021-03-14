# Base image
FROM python:3.8.5

# Copying requirements.txt file
COPY requirements.txt requirements.txt

# pip install 
RUN pip install --no-cache -r requirements.txt
RUN pip install jupyter

# Exposing ports
EXPOSE 8888

# Running jupyter notebook
# --NotebookApp.token ='TOKEN' is the password
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='TOKEN'"]
