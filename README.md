# ATM Supply Optimizer
A platform aiming to optimize the ATM supply planning for a set of bank ATMs that need to be supplied by a specific provider. The platform is solving the mTSP and CVRP, allowing the user to plan a set of efficient paths in a provided graph.

## Installation

### Requirements

The following software is required:

* Docker ( optionally )
* Python V3.12
* Pipenv ( if Docker is not used )

### Configuration 

After downloading the source code from git you will need to configure some parameters in the .env file.
The parameters are the following:

* DATASTORE_SUPPLY_INFO : The filepath of the ATM supply info in xlsx format.
* DATASTORE_SUPPLY_TEST_INFO : The filepath of the ATM supply info for evaluation in xlsx format.
* DATASTORE_OUTAGE_INFO : The filepath of the ATM outage info.

If you do install the predictor using Docker then you will need to setup a local MongoDB and provide the following parameters:

* MONGO_USERNAME : The MongoDB username.
* MONGO_PASSWORD : The MongoDB password.
* MONGO_DATABASE : The MongoDB database.
* MONGO_AUTH_DB : The MongoDB authentication database.
* MONGO_HOST : The MongoDB host.
* MONGO_PORT : The MongoDB port.

### Docker
The recomented way to install and use ATM_cash_deposit_predictor is through Docker compose. The instruction for installation are the following:

* Download the source code using git.
* Change directory to the root folder of the source.
* Execute the command `docker compose up --build -d`. It may need `sudo` rights depending on system setup.
* After the container is running you can either connect to the `atmo-python` container using the command `docker exec -it atmo-python /bin/bash` or you can run a command directly using `docker exec atmo-python <command>`

### Pipenv
If Docker is not available or not preferred Pipenv can also be used. The instruction for installation are the following:

* Download the source code using git.
* Change directory to the root folder of the source.
* Execute the command `pipenv install --python 3.12`.
* After the pipenv is created you can use the pipenv shell using the command `pipenv shell` and execute the commands normally or execute them using `pipenv run <command>`.

## Usage

The following commands are available:
* python main.py import_data : imports the data from the datastore files and add them to the MongoDB as defined in .env file.
* python main.py create_plans --datafile=”path/to/input/data” --model=ilp : run the prediction model, defined with the --model parameter, on the provided dataset, defined using the --data argument, and print a list of ATMs that will need supply tommorow.
* python main.py evaluate : evaluates the supported models using the test dataset defined in .env and prints various statistics.



## Licence

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
