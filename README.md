# MBI slabs

#### Installing the conda environment

I have included a yml file in the repository which I used for running my code. You can install this conda environment using

```conda
	conda env create -f env.yml
``` 

#### Config file 

The config file that you should use for generating the mock data is included [here](https://github.com/Supranta/mbi_slabs/blob/main/config/Alex/test.yaml)

#### Generating mock lognormal data

You can run the following code to create a mock lognormal data

```python
	python create_mock_data.py ./config/Alex/test.yaml
``` 
