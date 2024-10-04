# Variational Autoencoder (VAE)

A VAE implementation in Java that can learn from the column of an input table and produces a model that can estimate the reconstruction probability of the dataset itself or of datasets containing similar information.

## Preparation

Create a standalone uber-JAR containing all dependencies from the GIT repository or download the [pre-coocked JAR file](https://data.d4science.org/shub/E_RmlXSjJSbFVhZmVyT25YTFJJYlY1a3BJRWc0T0xueUVIOWNXamR3dStNV3RMZDl2WThJRE5rckY0b1cwWVU1Kw==).
To create the uber-Jar execute:

    ant
from the project folder (you need to install [ANT](https://ant.apache.org/manual/install.html)).

## General parameters
    -i: input file path
    -v: variable names (columns) separated by commas
    -h: number of hidden nodes
    -e: number of epochs
    -o: output folder
    -r: number of reconstruction samples
    -m: trained model file (for projections)
    -t: training mode active (true/false)
    

## How to train the model

1 - Prepare a table as a CSV file. See [Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv](https://github.com/cybprojects65/VariationalAutoencoder/blob/main/Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv)  as an example.
2 - Execute the Jar file as follows:

> java -cp vae.jar it.cnr.anomaly.JavaVAE
> -i"./Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv"
> -v"environment 2017_land_distance, environment 2017_mean_depth" -h5 -e1000 -o"./out/" -r16 -ttrue

3 - Retrieve the output as a CSV file in the "out" folder along with the model file (.bin) and the accessory files. This table contains the reconstruction probability for each input row and the classification in the 1st to 4th quantile to cluster small, medium, medium-high, and high values.


## How to test the model

1 - Prepare a table as a CSV file. See [Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv](https://github.com/cybprojects65/VariationalAutoencoder/blob/main/Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv)  as an example.
2 - Locate the model file (.bin)

3 - Execute the Jar file as follows:

> java -cp vae.jar it.cnr.anomaly.JavaVAE
> -i"./Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv"
> -v"environment 2017_land_distance, environment 2017_mean_depth" -o"./out/" -r16 -tfalse -m"./out/model.bin"

4 - Retrieve the output as a CSV file in the "out" folder. This table contains the reconstruction probability for each input row and the classification in the 1st to 4th quantile to cluster small, medium, medium-high, and high values.


## Docker version

A docker version is [also available on the Docker Hub](https://hub.docker.com/repository/docker/gianpaolocoro/variationalautoencoder/general).
