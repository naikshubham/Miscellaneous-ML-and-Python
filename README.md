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

- Notebook : [notebooks/dict_to_dataframe]
- Reference : https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi

### Writing a Decorator in Python
Decorator allows us to augment an existing function with extra code, without requiring us to change the existing function's code.

**4 things we need to understand to write a decorator**
- How to create a function
- How to pass a function as an argument to another function
- How to return a function from a function
- How to process any number and type of function arguments

### Mathematical transforms
- We can apply multiple mathematical transforms on numerical features like log(x), sqrt(x), apply polynomial transforms like (x2, x3, x4) or trignometric transformations like sin(x), cos(x) or tan(x).
- Question is what is the best transform. Its very problem specific, and we need to have domain knoweledge. If feature x has **powerlaw distribution**, taking something like log(x) makes sense because it converts it roughly into powerlaw distribution. Not always but close to a gaussian distribution.

<img src="data/powerLawDist2.jpg" width="350" title="Power Law distribution">

- A power law distribution looks like above with a long fat tail.
- Log is one of the cases of a box-cox transform.
- Logistic Regression is nothing but gaussian naives bayes with a bernoulli distribution on yi, which works very well if our features are gaussian distributed.

### Model specific featurizations
- Logistic regression assumes that the features are Gaussian distributed. If feature is power law distributed, it makes sense to transform feature f1 into log(f1).
- If we have features which have linear relationships i.e y=f1-f2+2f3 ,and we know this by domain knowledge. Then decision Trees can't really capture this relationship exactly.
- When we know that the interactions matter more in predicting dependent variables like interaction between f1 & f2 i.e to predict gender height & weight interact well we can use Random Forest or DT or GBDT.
- If we have text data and if we use Bag of words, linear models tend to perform very well. Because when we do BOW we end up having very high dimensional space. Bcz we can easily separate hyperplanes the +ve and -ve hyperplanes.

### Feature orthogonality
- The more different/orthogonal the features are the better the model is.
- When engineering new features we should take care that they are not correlated with each other.
- If we engineer a new feature f4 which is correlated with the error(ei) = (yi - yi^). Then adding this feature will boost the model performance, which actually is similar to Gradient boosting DT.

### Productionizing ML models
- To acheive low latency in production we can convert the logistic regression weights into a hash table or dictionaries. And simply read the weights from there and apply sigmoid. **sigmoid(Wi, Xi) = yi**.
- In case, of Decision Trees or Random forest, we can store the if else statements and execute them in C/C++/Python/Java.












