import random
import os

from typing import List

heatmap_classes = {
    'r3d_augment': ['openings'],
    'cubicasa5k_augment': ['doors', 'windows'],
    'multi_plans_augment': ['doors', 'sliding_doors', 'windows'],
    'cubicasa5k': ['doors', 'windows'],
}


def split_dataset(dataset, input_file: str = 'input.png', data_dir: str = 'annotations/hdd/',
                  output_dir: str = './', split_ration: float = 0.6, filtering: List = None,
                  test_only: bool = False, subfolders_only: bool = False, heatmaps: List = None,
                  k_fold: int = 0) -> List[str]:
    """
    Function for creation tran, val, test sets descriptions

    :param dataset: Name of new dataset
    :param input_file: Filename used for processing as input
    :param data_dir: Folder with input data
    :param output_dir: Folder where create description
    :param split_ration: Ration between train and (val, test), e.g. 0.6 means train sets includes 0.6% of all data
    :param filtering: Subset of folders which will be use
    :param test_only: If true - All data will be in test subset
    :param subfolders_only: If true - Not create description, only returns subfolders
    :param k_fold: Number of folds for cross validation (0 disables k-fold)

    :return: returns list of subfolders in dataset
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if heatmaps is None:
        heatmaps = heatmap_classes.get(dataset, [])

    test_sub_dirs = []
    if dataset in ['cubicasa', 'cubicasa5k']:
        sub_dirs = []
        for file in ["train.txt", "val.txt"]:
            sub_dirs += open(os.path.join(data_dir, file), 'r').read().splitlines()
        sub_dirs = [d[1:] for d in sub_dirs]
        test_sub_dirs = open(os.path.join(data_dir, "test.txt"), 'r').read().splitlines()
        test_sub_dirs = [d[1:] for d in test_sub_dirs]
        if filtering:
            sub_dirs = [d for d in sub_dirs if d.split('/')[-2] in filtering]
            test_sub_dirs = [d for d in test_sub_dirs if d.split('/')[-2] in filtering]
            filtering = None
    else:
        sub_dirs = os.listdir(data_dir)
        
    if filtering:
        sub_dirs = [d for d in sub_dirs if d in filtering]

    all_sub_dirs = sub_dirs + test_sub_dirs
    if subfolders_only:
        print("Sub folders only: Dataset: {0} Total samples: {1}".format(dataset, len(all_sub_dirs)))
        return all_sub_dirs

    random.shuffle(sub_dirs)

    if test_only:
        # test_only means all data in sub_dirs becomes test
        test = sub_dirs
        train = []
        val = []
        dirs_to_write = [test]
        types_to_write = ['test']
        print("Total: {0}, test: {1}".format(len(test), len(test)))
    elif k_fold > 0:
        if not test_sub_dirs:
            n_train = int(split_ration * len(sub_dirs))
            n_val = int((len(sub_dirs) - n_train) / 2)
            n_test = len(sub_dirs) - n_train - n_val
            test = sub_dirs[n_train + n_val:]
            train_val = sub_dirs[:n_train + n_val]
        else:
            test = test_sub_dirs
            train_val = sub_dirs

        fold_size = len(train_val) // k_fold
        folds = []
        for k in range(k_fold):
            if k == k_fold - 1:
                folds.append(train_val[k * fold_size:])
            else:
                folds.append(train_val[k * fold_size:(k + 1) * fold_size])
        
        print(f"Total train+val: {len(train_val)}, test: {len(test)}, folds: {k_fold}")
        
        dirs_to_write = folds + [test]
        types_to_write = [f'fold_{k}' for k in range(k_fold)] + ['test']
    else:  # Create split: train/(val, test)
        if not test_sub_dirs:
            n_train = int(split_ration * len(sub_dirs))
            n_val = int((len(sub_dirs) - n_train) / 2)
            n_test = len(sub_dirs) - n_train - n_val
            
            train = sub_dirs[:n_train]
            val = sub_dirs[n_train:n_train + n_val]
            test = sub_dirs[n_train + n_val:]
        else:
            # Revert to standard 60/20/20 approximately, but keeping the test_sub_dirs as test
            n_train = int(0.75 * len(sub_dirs)) # standard cubicasa split roughly
            n_val = len(sub_dirs) - n_train
            n_test = len(test_sub_dirs)
            
            train = sub_dirs[:n_train]
            val = sub_dirs[n_train:]
            test = test_sub_dirs
            
        print("Total: {0}, train: {1}, val: {2}, test: {3}".format(len(sub_dirs)+len(test_sub_dirs), len(train), len(val), len(test)))
        
        dirs_to_write = [train, val, test]
        types_to_write = ['train', 'val', 'test']

    for dir_list, split_type in zip(dirs_to_write, types_to_write):
        f = open(os.path.join(output_dir, dataset + '_' + split_type + '.txt'), 'w')
        for i in dir_list:
            input = os.path.join(dataset, i, input_file)
            mask = os.path.join(dataset, i, 'mask.png')
            line = input + '\t' + mask
            for h in heatmaps:
                line += '\t' + os.path.join(dataset, i, 'heatmap_' + h + '.png')
            f.write(line + '\n')
        f.close()
    return all_sub_dirs


if __name__ == '__main__':
    # split_dataset('cubicasa5k_single_augment')
    split_dataset('cubicasa5k_augment')
    # split_dataset('multi_plans', 'image.jpg')
    # split_dataset('multi_plans_augment')
    # split_dataset('multi_plans_test', 'image.jpg', True)
    # split_dataset('multi_plans_test_augment', test_only=True)
    split_dataset('r3d_augment')
