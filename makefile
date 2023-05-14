# create a new virutal environment
# once you create it you will need to switch to it
# WSL: source ./venv/Scripts/activate
create-venv:
	python -m venv venv

# install requirements into venv once you have switched to it
install-requirements:
	pip install -r requirements.txt


