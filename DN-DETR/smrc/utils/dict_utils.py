
def display_dict(dict_to_display):
    print('========================================')
    for key, value in dict_to_display.items():
        print(f'Key={key}, value={value}')


def sort_dict_by_value_len(my_dict, reverse=True):
    return sorted(my_dict.items(), key=lambda p:len(p[1]), reverse=reverse)


def display_dict_sorted_by_value(my_dict, reverse=True):
    print('======================================== Display dict sorted by values')
    for k,v in sort_dict_by_value_len(my_dict, reverse=reverse):
        print(f'Key={k}, value={v}')


def extract_sub_dict_by_key(my_dict, keys_to_keep):
    result_dict = {}
    for k, v in my_dict.items():
        if k in keys_to_keep:
            result_dict[k] = v
    return result_dict

