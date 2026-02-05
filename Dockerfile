#FROM registry.access.redhat.com/ubi8/python-39
FROM python:3.11-slim

# Source - https://stackoverflow.com/questions/71753536/how-to-deal-with-permission-errors-with-pip-inside-a-docker-compose-container
# Posted by Lindsay-Needs-Sleep
# Retrieved 05/11/2025, License - CC-BY-SA 4.0

# User root user to create a non-root user so we can install pip packages without root
USER root

# (fyi) You can pass args from docker-compose.yml, just remove the "=myuser" from here
ARG USERNAME=myuser

# for whatever reason the /home/username directory is not created with useradd  for me :/
# RUN useradd -u ${USER_UID} --gid ${USER_GID} ${USERNAME}
RUN adduser --uid 1001 --disabled-password ${USERNAME}

# Switch to our non-root user
USER ${USERNAME}
# add the default pip bin install location to the PATH
ENV PATH="$PATH:/home/${USERNAME}/.local/bin"

# Run your pip install commands

# Set working directory
WORKDIR /app

# Add application sources with correct permissions for OpenShift
USER 0
ADD app-src .

#RUN chown -R 1001:0 ./
RUN chown -R 1001:0 /app && \
    chmod -R g=u /app


USER 1001


COPY ./app-src/ ./

#RUN pip install -U "pip>=19.3.1"
#RUN pip3 install -r requirements.txt
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt"


COPY . .
ENV FLASK_APP=app
EXPOSE 5000

CMD python app.py runserver 0.0.0.0:5000