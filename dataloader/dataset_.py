import multiprocessing
import numpy as np
import tensorflow as tf


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)
    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             augment=False,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    if labels is None:
        memory_data = img_paths
    else:
        labels = tf.constant(labels)
        memory_data = (img_paths, labels)

    def augment_image(img):
        weak_aug = img
        if(np.random.randint(0, 2) == 1):
            weak_aug = tf.keras.layers.GaussianNoise(10)(weak_aug)
        strong_aug = img

        # Generate seed for augmentation functions in tensorflow
        seed = np.random.randint(1000)
        strong_aug = tf.image.random_jpeg_quality(strong_aug, 0, 10, seed=seed)
        strong_aug = tf.image.random_brightness(strong_aug, max_delta=0.2, seed=seed)
        strong_aug = tf.image.random_saturation(strong_aug, 0.2, 0.5, seed=seed)
        strong_aug = tf.keras.layers.GaussianNoise(10)(strong_aug)

        return weak_aug, strong_aug

    def parse_fn_with_label(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3

        if(augment):
            weak_aug, strong_aug = augment_image(img)
            weak_aug = map_fn(weak_aug)
            strong_aug = map_fn(strong_aug)

            return weak_aug, strong_aug, label

        img = map_fn(img)
        return img, label

    def parse_fn(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3
        
        if(augment):
            weak_aug, strong_aug = augment_image(img)
            weak_aug = map_fn(weak_aug)
            strong_aug = map_fn(strong_aug)

            return weak_aug, strong_aug

        img = map_fn(img)
        return img

    if map_fn and labels is None:  # fuse `map_fn` and `parse_fn`
        def map_fn_(path):
            return parse_fn(path)
    elif map_fn and labels is not None:
        def map_fn_(path, label):
            return parse_fn_with_label(path, label)# map_fn(*parse_fn_with_label(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return dataset

