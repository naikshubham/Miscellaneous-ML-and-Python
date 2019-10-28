# Miscellaneous-ML-and-Python

## The solution to the problems which I encounter while solving AI/ML/Python usecases.

### Parsing a large json file
- Parsing a large json file which doesn't fit into memory is difficult.
- It can be done using a package called **`ijson`**. ijson will iteratively parse the json file instead of reading it all at once. This is slower than directly reading the whole file in, but it enables us to work with large files that can't fit in memory.