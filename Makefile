init:
	pip install -r requirements.txt

run-api:
	fastapi run ./transcribe-ai/http/api.py

run-dev:
	fastapi dev ./transcribe-ai/http/api.py

.PHONY: init