# Contact information detection
Prediction whether there is contact information in the advertisement on a 
russian website.

## Task

You need to predict whether there is contact information in the website 
advertisement. You have the following fields for training:

* `title`,
* `description`,
* `subcategory`,
* `category`,
* `price`,
* `region`,
* `city`,
* `datetime_submitted`.

Target: `is_bad`

The data is available at the following link: [link](
https://drive.google.com/file/d/1IOo206jH0cbIMsWRfIZXwCeCAqysZLD3/view?usp=sharing
)

In the dataset incorrect labels may occur.

The output file of the model's results should be in csv format with the 
following columns:

* `index`: `int`, the position of the record in the file.
* `prediction`: `float` ranging from 0 to 1, representing the probability of 
* contact information being present in the advertisement.

|index  |prediction|
|-------|----------|
|0|0.12|
|1|0.95|
|...|...|
|N|0.68|

As the performance metric for the model is the averaged ROC-AUC for each 
advertisement category.


## Solution Execution

You can find the final solution in 
```lib/code_for_learning/solution.py``` to run this solution, run the following
steps.
1. Put `train.csv` and `test.csv` to the `nlp-rus-data` folder in the 
current directory
2. Build the Docker image using the command:\
```docker build -t nlp-rus -f Dockerfile . ```
3. Then run the container using the command:\
```docker run -it -v ~/nlp-rus/nlp-rus-data:/nlp-rus-data nlp-rus python lib/run.py ```
4. To load the final result back to your local computer: 
   1. check id of your container. You can find it in the Docker app or run this 
   command  \
   ```docker ps --no-trunc -a```
   2. load file with using this command\
   ```docker cp <container_id>:./src/nlp-rus-data .```

Container resources:
* 4 GB of RAM 
* 2 CPU cores 
* No GPU

Time limit:
* Processing 100,000 objects should not exceed 90 minutes.

## Baseline
The current baseline, which it is desirable to surpass, is 0.9.