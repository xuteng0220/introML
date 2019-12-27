house price


target
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

feature
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale











1.select features manually
1.1 feature selection automatically?
2.feature engineering
3.feature scaling
4.reduce dimensinality
5.regression


# load data
train = pd.read_csv('hp_train.csv')
test = pd.read_csv('hp_test.csv')

# select features manually
train_features = train.loc[:, ['LotArea','Neighborhood','OverallQual','OverallCond','ExterQual','BsmtQual','BsmtFinSF1','TotalBsmtSF','Electrical','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','BedroomAbvGr','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','MoSold']]

test_features = test.loc[:, ['LotArea','Neighborhood','OverallQual','OverallCond','ExterQual','BsmtQual','BsmtFinSF1','TotalBsmtSF','Electrical','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','BedroomAbvGr','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','MoSold']]

# length of train dataset
train_len = len(train_features)

# concatenate train and test data vertically
train_test = pd.concat([train_features, test_features], axis=0)

# get dummy variables
train_test_dummy = pd.get_dummies(train_test)

train_dummy = train_test_dummy.iloc[range(train_len+1), :].values
test_dummy = train_test_dummy.iloc[range(train_len+1, len(train_test_dummy)), :].values
y_train = train['SalePrice'].values

# import linear models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


lr = LinearRegression().fit(train_dummy, y_train)
lr.score(train_dummy, y_train)
lr.predict(test_dummy)

ridge = Ridge().fit(train_dummy, y_train)
ridge.score(train_dummy, y_train)
ridge.predict(test_dummy)

lasso = Lasso().fit(train_dummy, y_train)
lasso.score(train_dummy, y_train)
lasso.predict(test_dummy)

# import data reduceing models
from sklearn.decomposition import PCA

# import data scaling models
import sklearn. import StandardScale



















