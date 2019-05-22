#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from pprint import pprint


# We import the information from the Excel file

# In[51]:


table = pd.read_excel('phones.xlsx')
table.index = table['ID']
table = table.drop('ID', axis=1)


# In[52]:


table


# ### We create a dictionnary to easily map from the criteria to their respective weights.

# In[128]:


criteria = table.columns
weights  = [0.3, 0.1, 0.2, 0.15, 0.25]

if len(criteria) == len(weights) and np.isclose(1, sum(weight for weight in weights)):
    w_criteria = {criterion:weight for criterion, weight in zip(criteria, weights)}
else:
    print(f'Number of criteria: {len(criteria)}, number of weights: {len(weights)}')
    print(f'Sum of weights: {sum(weight for weight in weights)}')
    w_criteria = {criterion:0 for criterion in criteria}
    raise Exception(f'A weight is needed for each criterion and the sum of weights must be equal to one!')

w_criteria


# ### We create a dictionary to access the optimization direction (min or max) for each criterion

# In[131]:


senses = [0, 1, 1, 1, 1] # O and 1 because they automatically map to complementary bool values. 

if len(senses) == len(criteria):
    s_criteria = {criterion:sense for criterion, sense in zip(criteria, senses)}
else:
    raise Exception(f'Specify a value (0 for min, 1 for max) for each one of the criteria : {list(criteria)}')


# ## Create the normalised decision matrix

# For the normalisation part, there are many possible rules. The following options are available on this implementation: 
# $$ x_{ij} \;\; = \;\; \frac{a_{ij}}{\sqrt{\sum_{i}^{N} a_{ij}}}$$

# In[62]:


n_table = table.copy()

sq_sum_squares = table.apply(lambda y: np.sqrt(sum(x**2 for x in y)))
sq_sum_squares = dict(sq_sum_squares)

for column in table.columns:
    f = (lambda y: lambda x: x/sq_sum_squares[y])(column) 
    n_table[column] = table[column].map(f)
    
n_table.head(5)


# # EXPLORE MORE RULES !

# 

# ### Create the weighted normalised decision matrix

# In[63]:


w_n_table = n_table.copy()

for column in n_table.columns: 
    w_n_table[column] = n_table[column].map(lambda x: x*w_criteria[column])
    
w_n_table.head(5)


# ### Computation of the concordance matrix

# In[93]:


concordance_matrix = pd.DataFrame(columns=table.index, index=table.index)

for phone in w_n_table.index:
    for phone2 in w_n_table.index:
        _sum = 0
        for criterion in w_n_table.columns:
            if w_n_table.loc[phone, criterion] >= w_n_table.loc[phone2, criterion]:
                _sum += w_criteria[criterion]
        if phone == phone2:
            concordance_matrix.loc[phone, phone2] = 0
        else:
            concordance_matrix.loc[phone, phone2] = _sum
        
concordance_matrix


# ### Binary concordance set

# In[106]:


binary_concordance_matrix = concordance_matrix.copy()

sum_of_sums_of_columns = sum(concordance_matrix[column].sum() for column in concordance_matrix.columns)
non_diagonal_entries = concordance_matrix.shape[0]**2 - concordance_matrix.shape[0]
c_bar = sum_of_sums_of_columns / non_diagonal_entries

binary_concordance_matrix = concordance_matrix.applymap(lambda x: 1 if x > c_bar else 0)

binary_concordance_matrix


# In[ ]:


negatives = lambda y: list(filter(lambda x: True if x < 0 else False, y))
positives = lambda y: list(filter(lambda x: True if x > 0 else False, y))


# In[117]:





# In[116]:





# In[ ]:




