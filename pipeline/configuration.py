import os
import jsonpickle

# add your source paths here
path_to_dataset=""
path_to_labels=""
path_to_output_folder=""

# creates a new interactive config object/file for the training pipeline
class Configuration:
    def __init__(self, config_name, path_to_dataset, path_to_labels, path_to_output_folder):

        self.path_to_dataset = path_to_dataset
        self.path_to_labels = path_to_labels
        self.path_to_output_folder = path_to_output_folder
        self.path_to_results = os.path.join(path_to_output_folder, "results")
        self.config_name = config_name

        self.syn_results = ""
        self.syn_data_generation_path = ""
        self.path_to_syn_run_folder = ""
        self.syn_train_path = ""
        self.tmp_dataset_path = ""
        self.tmp_label_path = ""
        self.syn_train_data = ""
        self.syn_train_results = ""
        self.generated_samples_path = ""
        self.fusioned_syn_samples_path = ""
        self.fusioned_syn_labels_path = ""
        self.fusioned_syn_labels_control_path = ""
        self.fusioned_syn_samples_control_path = ""
        self.control_samples_path = ""
        self.syn_anomaly_transformations = {}
        self.cropped_control_data = ""
        self.train_data_control = ""
        self.csv_datapath = ""
        self.csv_test_datapath = ""
        self.detection_img_train_data = ""
        self.detection_seg_train_data = ""
        self.detection_training_path = ""
        self.path_to_matching_infos = ""

        self.write_config_file(self.config_name)



    def write_config_file(self, config_name=None):
        json_string = jsonpickle.encode(self, indent=0)
        if config_name is not None:
            path = os.path.join(os.path.join(".", "configs"), config_name + ".json")
            self.config_name = config_name
            try:
                with open(path, 'x', encoding='utf-8') as fi:
                    fi.write(json_string)
            except FileExistsError:
                with open(path, 'w', encoding='utf-8') as fi:
                    fi.write(json_string)
        else:
            path = os.path.join(os.path.join(".", "configs"), self.config_name + ".json")
            os.chmod(path, 0o644)
            with open(path, 'w', encoding='utf-8') as fi:
                fi.write(json_string)

    def add_anomaly_transformation(self, name, label, scale_factor, centroid, shape):
        self.syn_anomaly_transformations[name] = {
            "label": label,
            "scale_factor": scale_factor,
            "centroid": centroid,
            "shape": shape
        }

    def print_config(self):
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))


def load_config_file(config_name):
    path = os.path.join(os.path.join(".", "configs"), config_name + ".json")

    with open(path, 'r', encoding='utf-8') as fi:
        return jsonpickle.decode(fi.read())


def create_home_configuration(config_name):
    config = Configuration(config_name, path_to_dataset,
                           path_to_labels,
                           path_to_output_folder)
    config.write_config_file(config_name)
