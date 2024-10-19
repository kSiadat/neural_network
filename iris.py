from ucimlrepo import fetch_ucirepo


iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets
z = iris.data

# metadata
#print(iris.metadata)

# variable information
#print(iris.variables)

print(z.features.tolist())
