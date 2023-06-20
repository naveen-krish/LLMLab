# Use the official Python base image
FROM python:3.9.6

#CMD mkdir -p /app
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip3 install  -r requirements.txt

EXPOSE 8501
# Copy the rest of the application code to the working directory
COPY . .

# Expose port 8501 (change it to the appropriate port for your application)

# Set the entrypoint to 'streamlit'
ENTRYPOINT ["streamlit","run"]

# Set the default command for the entrypoint
CMD ["langchainbot.py"]

#ENTRYPOINT ["streamlit", "run", ".py", "--server.port=8501", "--server.address=0.0.0.0"]
