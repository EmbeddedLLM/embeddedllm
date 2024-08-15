# Benchmark

## Benchmark a Model
To benchmark a model, run this
* --backend `cpu` | `ipex` | `openvino` | `directml`
* --model_name `Name of the Model`
* --model_path `Path to Model` | `Model Repo ID`
* --token_in `Number of Input Tokens (Max 2048)`
* --token_out `Number of Output Tokens`

```shell
python ellm_benchmark.py --backend <cpu | ipex | openvino | directml> --model_name <Name of the Model> --model_path <Path to Model | Model Repo ID> --token_in <Number of Input Tokens (Max 2048)> --token_out <Number of Output Tokens>
```


## Loop to benchmark the models
Customise your benchmarking config
```python
# Define the models
model_names = [
    # model names

]

# Define the model paths
model_paths = [
    # path to model in order to model names / model repo id

]

# Define the token length
token_in_out = [
    (1024, 1024),
    (1024, 512),
    (1024, 256),
    (1024, 128),
    (512, 1024),
    (512, 512),
    (512, 256),
    (512, 128),
    (256, 1024),
    (256, 512),
    (256, 256),
    (256, 128),
    (128, 1024),
    (128, 512),
    (128, 256),
    (128, 128),
]

# Choose backend
backend = "cpu"
backend = "directml"
backend = "ipex"
backend = "openvino"

# Number of loops
loop_count = 20
```
```shell
python loop_ellm_benchmark.py
```

## Generate a Report (`XLSX`) of a Model's Benchmark
To Generate report for a model, run this
* --model_name `Name of the Model`
```shell
python analyse_detailed_benchmark.py --model_name <Name of the Model>
```

## Generate Reports (`XLSX`) of Models' Benchmark
List out the models that you want to have report of benchmarking
```python
model_names = [
    # model names
    
]
```
```shell
python loop_analyse_detailed_benchmark.py
```