from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample dataset (Boolean values)
data = {
    'Bread': [True, True, False, True],
    'Milk': [True, False, False, True],
    'Beer': [False, True, True, True],
    'Diapers': [False, True, True, True],
    'Eggs': [True, False, True, False]
}
df = pd.DataFrame(data)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Pass num_itemsets to association_rules
rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric="lift", min_threshold=1.0)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
