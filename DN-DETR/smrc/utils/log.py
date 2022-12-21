from .load_save import save_1d_list_to_file


def write_exception_log(file_path, msg):
    save_1d_list_to_file(file_path=file_path, list_to_save=[msg])

