.PHONY: help update

ENV_DIST=jax
CONDA_DIST=mamba
OS=macos

help:
	@echo "The following make targets are available:"
	@echo "	create			install all dependencies for environment with conda/mamba "
	@echo "	update			update all dependencies for environment with conda/mamba "
	@echo ""
	@echo "Environments:"
	@echo "	ENV_DIST 		jax/pytorch "
	@echo "	CONDA_DIST 		conda/mamba "
	@echo "	OS				macos "
	@echo ""
	@echo "Example commands:"
	@echo "	make create CONDA_DIST=conda ENV_DIST=pytorch OS_DIST=macos"
	@echo "	make update CONDA_DIST=mamba ENV_DIST=jax OS_DIST=macos"

create:
	$(CONDA_DIST) env create -f environments/environment_$(ENV_DIST)_$(OS).yml

update:
	$(CONDA_DIST) env update -f environments/environment_$(ENV_DIST)_$(OS).yml


# create-pytorch:
# 	conda env create -f environments/environment_pytorch_macos.yml

# update-pytorch:
# 	conda env update -f environments/environment_pytorch_macos.yml
