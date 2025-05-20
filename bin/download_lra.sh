mkdir raw_datasets

# Clone and unpack the LRA object.
# This can take a long time, so get comfortable.
rm -rf lru/raw_datasets/lra_release.gz lru/raw_datasets/lra_release  # Clean out any old datasets.
wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P lru/raw_datasets

# Add a progress bar because this can be slow.
pv lru/raw_datasets/lra_release.gz | tar -zx -C lru/raw_datasets/
