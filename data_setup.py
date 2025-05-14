'''Download CIFAR dataset and store them as numpy files'''

def save_as_numpy(dataset, prefix):
    # convert each raw image into numpy array, then stack all of them together into one numpy array
    image_data = np.stack([np.array(img) for img, _ in dataset], axis = 0)
    # convert all raw labels into numpy array
    label_data = np.array([label for _, label in dataset])
    
    # store the images and labels based on the prefix
    np.save(f'{prefix}_images.npy', image_data)
    np.save(f'{prefix}_labels.npy', label_data)
    
    print(f'{prefix} data is stored as {prefix}_images.npy, {prefix}_labels.npy')

def get_store_data():
    # download and store raw training and testing data

    raw_train = torchvision.datasets.CIFAR10(root = './cifar10_dataset', train = True, download = True)
    raw_test = torchvision.datasets.CIFAR10(root = './cifar10_dataset', train = False, download = True)
    
    # save the dataset as numpy file

    save_as_numpy(raw_train, 'train')
    save_as_numpy(raw_test, 'test')
