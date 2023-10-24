SHELL := /bin/bash
PYTHON_LIBS = word2vec script

# Function to print a colored header
COL=\033[1;35m
NC=\033[0m
define header
    @echo -e "${COL}$1${NC}"
endef

# Set up venv
setup:
	$(call header,"[make setup]")
	python3 -m venv .venv
	source ./.venv/bin/activate && \
		python3 -m pip install pip==23.1.2

# Install needed packages to run code
install:
	$(call header,"[make install]")
	python3 -m pip install -r requirements.txt -r requirements-dev.txt

clean:
	$(call header,"[make clean]")
	rm -rf .venv/

static_type_check:
	$(call header,"[static_type_check]")
	python3 -m mypy $(PYTHON_LIBS) --config-file mypy.ini
