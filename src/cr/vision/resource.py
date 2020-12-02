"""
Resource
"""




def get(file_name, origin, 
    file_hash=None,
    cache_subdir=None,
    extract=False,
    cache_dir=None):
    if cache_dir is None:
        home_dir = Path.home()
        cache_dir = home_dir / '.cr-vision'
    data_dir = cache_dir 
    if cache_subdir is not None:
        data_dir = data_dir / cache_subdir
    # Make sure that the directory in which the file
    # will be downloaded exists
    cache_subdir.mkdir(parents=True, exist_ok=True)
    file_path = cache_subdir / file_name
    print('Downloading data from', origin)
    
    pass

