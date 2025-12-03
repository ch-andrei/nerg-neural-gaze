def split_list(list, num_splits, append_leftover_to_last=False):
    """
    divides a list into N splits
    :param list:
    :param num_splits:
    :param append_leftover_to_last: add leftover tasks to last thread or distribute to all threads equally
    :return:
    """
    n = len(list)
    list_per_split = [[] for _ in range(num_splits)]
    num_per_split = int(len(list) / num_splits)
    leftover = n - num_per_split * num_splits
    for i in range(num_splits):
        starting_index = i * num_per_split
        for item in list[starting_index: starting_index + num_per_split]:
            list_per_split[i].append(item)
    starting_index = num_splits * num_per_split
    for i, item in enumerate(list[starting_index: starting_index + leftover]):
        if append_leftover_to_last:
            index = num_splits - 1
        else:
            index = i % num_splits
        list_per_split[index].append(item)
    return list_per_split
