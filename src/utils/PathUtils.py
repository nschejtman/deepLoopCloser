def firstParentWithNamePath(path: str, name: str):
    split_path = path.split('/')
    split_path.reverse()
    end_idx = -1
    for i, v in enumerate(split_path):
        if v == name:
            end_idx = i
            break
    split_path.reverse()
    parent_path = split_path[:end_idx + 1]
    return str.join("/", parent_path)
