FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN echo '#!/bin/bash\npython phi_server.py & \nuvicorn main:app --host 0.0.0.0 --port 3001' > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]

EXPOSE 8000 8001