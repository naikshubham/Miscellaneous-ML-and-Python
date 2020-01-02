# Miscellaneous-ML-and-Python

## The solution to the problems which I encounter while solving AI/ML/Python usecases.

### Parsing a large json file
- Parsing a large json file which doesn't fit into memory is difficult.
- It can be done using a package called **`ijson`**. ijson will iteratively parse the json file instead of reading it all at once. This is slower than directly reading the whole file in, but it enables us to work with large files that can't fit in memory.

### Constructing pandas DataFrame from a dictionary gives `“ValueError: If using all scalar values, you must pass an index”`
- The error message says that if we're passing scalar values, we have to pass an index. So we can either not use scalar values for the columns -- e.g. use a list:

```python
df = pd.DataFrame({'A': [a], 'B': [b]})
>>>df
   A  B
0  2  3
```
or use scalar values and pass an index:

```python
df = pd.DataFrame({'A': a, 'B': b}, index=[0])
>>>df
   A  B
0  2  3
```

Notebook : [notebooks/]dict_to_dataframe
Reference : https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi