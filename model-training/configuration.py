import os

root_data_directory = "C:\\Users\\seliv\\Documents\\Projects\\digit-recognition\\data"

train_data_file_name = "train.csv"
train_data_file_path = os.path.join(root_data_directory, train_data_file_name)

test_data_file_name = "test.csv"
test_data_file_path = os.path.join(root_data_directory, test_data_file_name)


validation_split_ratio = 0.8
