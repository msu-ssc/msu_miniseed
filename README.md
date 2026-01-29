> [!WARNING]  
> This whole thing is "vibe coded" by a guy who knows NOTHING about seismology.
> 
> This gets the strongest possible "Use at your own risk" disclaimer

# MSU miniSEED

> [!CAUTION]
> Seriously: Do not use this code for serious work. YOU HAVE BEEN WARNED.

Simple parsing/creation of [miniSEED 3](https://docs.fdsn.org/projects/miniseed3/en/latest/index.html) seismometer data files (including Steim-1 and Steim-2 compression/decompression) for use at the Morehead State University Space Science Center

The only purpose of this is for us to be able to check whether a `.miniseed` file that we receive from a spacecraft appears to be properly formatted before we send it on to some actual scientists.

## Usage

There are only a few things in the public API.

```py
import msu_miniseed


parsed_file: msu_miniseed.ParsedFile = msu_miniseed.parse_file("seismometer_data.miniseed")

print(f"Number of records: {len(parsed_file)}")

# `parsed_file.dataframe` will be a Pandas dataframe
print(f"Head data:")
print(parsed_file.dataframe.head())

# Can output to a .csv or a .miniseed
parsed_file.to_csv("seismometer_data.csv")
parsed_file.to_miniseed("seismometer_data2.miniseed")

# That CSV can also be loaded as a `ParsedFile`
parsed_file_2: msu_miniseed.ParsedFile = msu_miniseed.parse_csv(seismometer_data.csv)
print(f"Number of records (in reconstructed ParsedFile): {len(parsed_file_2)}")

# Getting the actual data.
# The actual instrument value is in the column "sample"
timestamp_values = parsed_file.dataframe["timestamp"].to_numpy()
sample_values = parsed_file.dataframe["sample"].to_numpy()

# These will be normal 1D numpy arrays. You can do whatever with them:
import matplotlib.pyplot as plt

plt.plot(timestamp_values, sample_values)
plt.show()
```

## Not maintained

This repo is not maintained or supported at all. Please do not create issues or contact the authors with questions. We probably don't know the answer, anyway.