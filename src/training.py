from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot
import os
import argparse
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
general_logs_path = "logs"
os.makedirs(general_logs_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(general_logs_path, "running_logs.log"), level=logging.INFO,
                    format=logging_str, filemode="a")



def training(config_path):
    config = read_config(config_path)
    logging.info(f"configuration data loaded successfully from {config_path}")

    validation_datasize = config["params"]["validation_datasize"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    logging.info("training, validation and test data loaded successfully")
    LOSSFUNCTION =  config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_ClASSES = config["params"]["num_classes"]

    model = create_model(LOSSFUNCTION, OPTIMIZER, METRICS, NUM_ClASSES)
    logging.info("created model successfully")

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, 
              validation_data=VALIDATION_SET, 
              epochs=EPOCHS)
    logging.info("ANN trained and validated successfully")

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)

    plot_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    plot_name = config["artifacts"]["plot_name"]
    loss_acc = history.history
    save_plot(loss_acc, plot_name, plot_dir_path)

    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yml")

    parsed_args = args.parse_args()
    CONFIG_PATH = parsed_args.config

    # config = read_config(CONFIG_PATH)
    # logs_dir = config["logs"]["logs_dir"]
    #
    # general_logs = config["logs"]["general_logs"]
    # tensorboard_logs = config["logs"]["tensorboard_logs"]
    #
    # general_logs_path = os.path.join(logs_dir, general_logs)
    # tensorboard_logs = os.path.join(logs_dir, tensorboard_logs)
    #
    # os.makedirs(general_logs_path, exist_ok=True)
    # os.makedirs(tensorboard_logs, exist_ok=True)


    logging.info("<<< ANN training started >>>")
    training(config_path=CONFIG_PATH)
    logging.info("<<< ANN training done >>>")
    logging.info("*"*80)


