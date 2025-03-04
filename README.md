**Disaggregation of energy consumption from a trained model against a Dataset containing only the Aggregate.**

In this project, a model is generated based on data from the OMPM measures. Subsequently, using this trained model (workbooks 1 to 7), we try to disaggregate the demand using only a dataset containing only the aggregate measures.
The last booklet, which is the most interesting one, shows the use of siteonlyapi, a new NILMTK interface which is a modification of ExperimentAPI of NILMTK and which allows NILMTK users to obtain the energy demands of their homes/buildings for different potential appliances starting from a trained dataset.


Let's start with the OMPM data to demonstrate the use of this API. This experiment shows how the user can convert his meter data into the appropriate DSUAL format and call the API to disaggregate energy into appliance demands based on the training set. The meter data is converted to the appropriate format. We also change the start and end dates of your test data set and also enter the values of the different parameters in the dictionary. Since we need several appliances, we enter the names of all required appliances in the parameter 'appliances'. We also mention site_only is true because we want to disaggregate the site meter data only without any comparison with the sub-meter data.

It is therefore crucial how applications, methods and datasets are typified, which we can see below:


experiment1 = {

`    `'power': {'mains': ['active'],'appliance': ['active']},

`    `'sample\_rate': 60,

`    `'appliances': [2,3,4,5,6],

`    `*#'appliances': ['Fryer', 'LED Lamp',  'Incandescent lamp','Laptop Computer', 'Fan'],*

`    `'methods': {"CO":**CO**({}),"FHMM":**FHMMExact**({'num\_of\_states':2}),'Mean':**Mean**({}),'Hart':**Hart85**({})},

`    `'site\_only' : True,

`  `'train': {    

`    `'datasets': {

`        `'DSUAL': {

`            `'path': 'ualm2.h5',

`            `'buildings': {

`                `1: {

`                    `'start\_time': '2023-02-24 14:47:10', 

`                    `'end\_time': '2023-02-24 20:03:54'

`                    `}

`                `}                

`            `}

`        `}

`    `},

`  `'test': {

`    `'datasets': {

`        `'DSUAL': {

`            `'path': 'ualm3.h5',

`            `'buildings': {

`                `1: {

`                    `'start\_time': '2023-05-13 17:22:02',

`                    `'end\_time': '2023-05-13 23:43:15'

`                    `}

`                `}

`            `}

`        `},

`        `'metrics':['rmse']

`    `}

}

**Explanation of the code**


The provided code initialises an **API object** using the provided experiment configuration. A breakdown of the experiment configuration is shown below.


-Power specification:** Mains and device power is specified as "active" power.
-Sampling frequency:** The sampling frequency is set to 60 seconds.
-Devices:** The experiment includes devices with identifiers 2, 3, 4, 5 and 6.
-Methods/Algorithms:** Four NILM algorithms are used: CO, FHMM, Mean and Hart. 
-Site-only mode:** This mode is enabled (site_only=True), which means that only the aggregated power at site level is taken into account for training and testing.
-Training dataset:** The training dataset is taken from the 'DSUAL' dataset, stored in the file 'ualm2.h5'. Data from building 1 is used, starting from '2023-02-24 14:47:10' and up to '2023-02-24 20:03:54'.
-Test data set:** The test data set comes from the data set 'DSUAL', stored in the file 'ualm3.h5'. The data from building 1, starting from '2023-05-13 17:22:02' and up to '2023-05-13 23:43:15' is used.
-Evaluation metric:** RMSE (root mean square error) is chosen as the evaluation metric.


The API object instantiated with this configuration will perform the NILM experiment using the specified algorithms and datasets, and evaluate its performance using RMSE


When api_results_experiment_1 = API(experiment1) is called with the contents of experiment1, the following API class functions are called:
1.**constructor (__init__):**
- The attributes of the API class are initialised with the parameters provided in experiment1.
- The experiment method is called to start the experiment.
2.**experiment method:**
- This method is called by the constructor.
- The classifiers (methods/algorithms) specified in experiment1['methods'] are iterated.
- Depending on whether the chunk training option is enabled or not, the train_chunk_wise or train_jointly methods are called to train the classifiers.
- After training, tests are performed either in chunks or together by calling the test_chunk_wise or test_jointly methods.
3.**Training and Testing Methods:** **Training and Testing Methods
- The training methods are called train_chunk_wise or train_jointly depending on whether training is done in chunks or together.
- The test methods test_chunk_wise or test_jointly are called depending on whether testing is done in chunks or together.
4.**Other Ancillary Methods:**
- Auxiliary methods such as dropna, store_classifier_instances, call_predict, predict, and compute_loss can be called indirectly within other methods mentioned above, depending on the implementation logic and execution flow.
In summary, when **API(experiment1)** is called, a number of methods within the **API** class are executed to perform the energy disaggregation experiment.






## API CONTENT ##


This is a Python class called API designed for NILM (Non Intrusive Load Monitoring) experiments. It allows you to train and test various disaggregation algorithms on a dataset of building electricity readings.

Here is a breakdown of the class:


**Initialization:**

- \_\_init\_\_(self, params): This function initialises the class with parameters like appliances, data sets, training methods, etc.

**Experiment Functions:**


- experiment (self): This function calls the training and testing functionalities....
- train\_chunk\_wise (self, clf, d, current\_epoch): This function trains the classifiers in a piecewise manner for better memory management...
- test\_chunk\_wise (self, d): This function tests the classifiers in a fragmented way...
- train\_jointly (self, clf, d): This function trains the classifiers on the entire dataset in a joint manner.
- test\_jointly (self, d): This function tests the classifiers on the entire dataset jointly.
**Data Handling Functions:**

- dropna(self, mains\_df, appliance\_dfs=[]):This function removes rows with missing values from the main stream and appliance data frames, maintaining consistency.
**Model Handling Functions:**

- store\_classifier\_instances(self):This function initialises the models based on the specified methods.
**Prediction and Evaluation Functions:**

- call\_predict (self, classifiers, timezone): This function calculates predictions on the test data using all models and compares them using specified metrics.
- predict (self, clf, test_elec, test_submeters, sample_period, timezone ): This function generates predictions for a specific classifier in the test data set.
- compute\_loss (self,gt,clf\_pred, loss\_function): This function computes the loss (error) between the ground truth and the predictions for each appliance.

**Other Functions:**


- display\_predictions (self): This function plots the ground truth and predicted energy consumption for each appliance.







**Python code**


From nilmtk.dataset import DataSet

from nilmtk.metergroup import MeterGroup

import **pandas** as **pd**

from nilmtk.losses import \*

import **numpy** as **np**

import **matplotlib**.**pyplot** as **plt**

import **datetime**

from **IPython**.**display** import **clear\_output**

class **API**():

`    `"""

`    `The API ia designed for rapid experimentation with NILM Algorithms.

`    `"""

`    `def **\_\_init\_\_**(self,params):

`        `"""

`        `Initialize the API with default parameters and then start the experiment.

`        `"""

`        `self.appliances = []

`        `self.train\_submeters = []

`        `self.train\_mains = **pd**.**DataFrame**()

`        `self.test\_submeters = []

`        `self.test\_mains = **pd**.**DataFrame**()

`        `self.gt\_overall = {}

`        `self.pred\_overall = {}

`        `self.classifiers=[]

`        `self.errors = []

`        `self.errors\_keys = []

`        `self.power = params['power']

`        `for elems in params['appliances']:

`            `self.appliances.**append**(elems)

`        `self.train\_datasets\_dict = params['train']['datasets']

`        `self.test\_datasets\_dict = params['test']['datasets']

`        `self.metrics = params['test']['metrics']

`        `self.methods = params['methods']

`        `self.sample\_period = params.get("sample\_rate", 1)

`        `self.artificial\_aggregate = params.get('artificial\_aggregate', False)

`        `self.chunk\_size = params.get('chunk\_size', None)

`        `self.display\_predictions = params.get('display\_predictions', False)

`        `self.DROP\_ALL\_NANS = params.get("DROP\_ALL\_NANS", True)

`        `self.site\_only = params.get('site\_only',False)

`        `self.**experiment**()



`    `def **experiment**(self):

`        `"""

`        `Calls the Experiments with the specified parameters

`        `"""

`        `self.**store\_classifier\_instances**()

`        `d=self.train\_datasets\_dict

`        `for model\_name, clf in self.classifiers:

`            `*# If the model is a neural net, it has an attribute n\_epochs, Ex: DAE, Seq2Point*

`            `**print** ("Started training for ",clf.MODEL\_NAME)

`            `*# If the model has the filename specified for loading the pretrained model, then we don't need to load training data*

`            `if **hasattr**(clf,'load\_model\_path'):

`                `if clf.load\_model\_path:

`                    `**print** (clf.MODEL\_NAME," is loading the pretrained model")

`                    `continue

`            `*# if user wants to train chunk wise*

`            `if self.chunk\_size:

`                `*# If the classifier supports chunk wise training*

`                `if clf.chunk\_wise\_training:

`                    `*# if it has an attribute n\_epochs. Ex: neural nets. Then it is trained chunk wise for every wise*

`                    `if **hasattr**(clf,'n\_epochs'):

`                        `n\_epochs = clf.n\_epochs

`                        `clf.n\_epochs = 1

`                    `else:

`                        `*# If it doesn't have the attribute n\_epochs, this is executed. Ex: Mean, Zero*

`                        `n\_epochs = 1

`                    `*# Training on those many chunks for those many epochs*

`                    `**print** ("Chunk wise training for ",clf.MODEL\_NAME)

`                    `for i in **range**(n\_epochs):

`                        `self.**train\_chunk\_wise**(clf, d, i)

`                `else:

`                    `**print** ("Joint training for ",clf.MODEL\_NAME)

`                    `self.**train\_jointly**(clf,d)            

`            `*# if it doesn't support chunk wise training*

`            `else:

`                `**print** ("Joint training for ",clf.MODEL\_NAME)

`                `self.**train\_jointly**(clf,d)            

`            `**print** ("Finished training for ",clf.MODEL\_NAME)

`            `**clear\_output**()

`        `d=self.test\_datasets\_dict

`        `if self.chunk\_size:

`            `**print** ("Chunk Wise Testing for all algorithms")

`            `*# It means that, predictions can also be done on chunks*

`            `self.**test\_chunk\_wise**(d)

`        `else:

`            `**print** ("Joint Testing for all algorithms")

`            `self.**test\_jointly**(d)

`    `def **train\_chunk\_wise**(self, clf, d, current\_epoch):

`        `"""

`        `This function loads the data from buildings and datasets with the specified chunk size and trains on each of them. 

`        `"""



`        `for dataset in d:

`            `*# Loading the dataset*

`            `**print**("Loading data for ",dataset, " dataset")          

`            `for building in d[dataset]['buildings']:

`                `*# Loading the building*

`                `train=DataSet(d[dataset]['path'])

`                `**print**("Loading building ... ",building)

`                `train.set\_window(start=d[dataset]['buildings'][building]['start\_time'],end=d[dataset]['buildings'][building]['end\_time'])

`                `mains\_iterator = train.buildings[building].elec.mains().load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type = self.power['mains'], sample\_period=self.sample\_period)

`                `appliance\_iterators = [train.buildings[building].elec[app\_name].load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type=self.power['appliance'], sample\_period=self.sample\_period) for app\_name in self.appliances]

`                `**print**(train.buildings[building].elec.mains())

`                `for chunk\_num,chunk in **enumerate** (train.buildings[building].elec.mains().load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type = self.power['mains'], sample\_period=self.sample\_period)):

`                    `*# Loading the chunk for the specifeid building*

`                    `*#Dummry loop for executing on outer level. Just for looping till end of a chunk*

`                    `**print**("Starting enumeration..........")

`                    `train\_df = **next**(mains\_iterator)

`                    `appliance\_readings = []

`                    `for i in appliance\_iterators:

`                        `try:

`                            `appliance\_df = **next**(i)

`                        `except **StopIteration**:

`                            `appliance\_df = **pd**.**DataFrame**()

`                        `appliance\_readings.**append**(appliance\_df)

`                    `if self.DROP\_ALL\_NANS:

`                        `train\_df, appliance\_readings = self.**dropna**(train\_df, appliance\_readings)



`                    `if self.artificial\_aggregate:

`                        `**print** ("Creating an Artificial Aggregate")

`                        `train\_df = **pd**.**DataFrame**(**np**.**zeros**(appliance\_readings[0].shape),index = appliance\_readings[0].index,columns=appliance\_readings[0].columns)

`                        `for app\_reading in appliance\_readings:

`                            `train\_df+=app\_reading

`                    `train\_appliances = []

`                    `for cnt,i in **enumerate**(appliance\_readings):

`                        `train\_appliances.**append**((self.appliances[cnt],[i]))

`                    `self.train\_mains = [train\_df]

`                    `self.train\_submeters = train\_appliances

`                    `clf.partial\_fit(self.train\_mains, self.train\_submeters, current\_epoch)



`        `**print**("...............Finished the Training Process ...................")

`    `def **test\_chunk\_wise**(self,d):

`        `**print**("...............Started  the Testing Process ...................")

`        `for dataset in d:

`            `**print**("Loading data for ",dataset, " dataset")

`            `for building in d[dataset]['buildings']:

`                `test=DataSet(d[dataset]['path'])

`                `test.set\_window(start=d[dataset]['buildings'][building]['start\_time'],end=d[dataset]['buildings'][building]['end\_time'])

`                `mains\_iterator = test.buildings[building].elec.mains().load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type = self.power['mains'], sample\_period=self.sample\_period)

`                `appliance\_iterators = [test.buildings[building].elec[app\_name].load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type=self.power['appliance'], sample\_period=self.sample\_period) for app\_name in self.appliances]

`                `for chunk\_num,chunk in **enumerate** (test.buildings[building].elec.mains().load(chunksize = self.chunk\_size, physical\_quantity='power', ac\_type = self.power['mains'], sample\_period=self.sample\_period)):

`                    `test\_df = **next**(mains\_iterator)

`                    `appliance\_readings = []

`                    `for i in appliance\_iterators:

`                        `try:

`                            `appliance\_df = **next**(i)

`                        `except **StopIteration**:

`                            `appliance\_df = **pd**.**DataFrame**()

`                        `appliance\_readings.**append**(appliance\_df)

`                    `if self.DROP\_ALL\_NANS:

`                        `test\_df, appliance\_readings = self.**dropna**(test\_df, appliance\_readings)

`                    `if self.artificial\_aggregate:

`                        `**print** ("Creating an Artificial Aggregate")

`                        `test\_df = **pd**.**DataFrame**(**np**.**zeros**(appliance\_readings[0].shape),index = appliance\_readings[0].index,columns=appliance\_readings[0].columns)

`                        `for app\_reading in appliance\_readings:

`                            `test\_df+=app\_reading

`                    `test\_appliances = []

`                    `for cnt,i in **enumerate**(appliance\_readings):

`                        `test\_appliances.**append**((self.appliances[cnt],[i]))

`                    `self.test\_mains = [test\_df]

`                    `self.test\_submeters = test\_appliances

`                    `**print**("Results for Dataset {dataset} Building {building} Chunk {chunk\_num}".**format**(dataset=dataset,building=building,chunk\_num=chunk\_num))

`                    `self.storing\_key = **str**(dataset) + "\_" + **str**(building) + "\_" + **str**(chunk\_num) 

`                    `self.**call\_predict**(self.classifiers, test.metadata['timezone'])

`    `def **train\_jointly**(self,clf,d):

`        `*# This function has a few issues, which should be addressed soon*

`        `**print**("............... Loading Data for training ...................")

`        `*# store the train\_main readings for all buildings*

`        `self.train\_mains = []

`        `self.train\_submeters = [[] for i in **range**(**len**(self.appliances))]

`        `for dataset in d:

`            `**print**("Loading data for ",dataset, " dataset")

`            `train=DataSet(d[dataset]['path'])

`            `for building in d[dataset]['buildings']:

`                `**print**("Loading building ... ",building)

`                `train.set\_window(start=d[dataset]['buildings'][building]['start\_time'],end=d[dataset]['buildings'][building]['end\_time'])

`                `train\_df = **next**(train.buildings[building].elec.mains().load(physical\_quantity='power', ac\_type=self.power['mains'], sample\_period=self.sample\_period))

`                `train\_df = train\_df[[**list**(train\_df.columns)[0]]]

`                `appliance\_readings = []



`                `for appliance\_name in self.appliances:

`                    `appliance\_df = **next**(train.buildings[building].elec[appliance\_name].load(physical\_quantity='power', ac\_type=self.power['appliance'], sample\_period=self.sample\_period))

`                    `appliance\_df = appliance\_df[[**list**(appliance\_df.columns)[0]]]

`                    `appliance\_readings.**append**(appliance\_df)

`                `if self.DROP\_ALL\_NANS:

`                    `train\_df, appliance\_readings = self.**dropna**(train\_df, appliance\_readings)

`                `if self.artificial\_aggregate:

`                    `**print** ("Creating an Artificial Aggregate")

`                    `train\_df = **pd**.**DataFrame**(**np**.**zeros**(appliance\_readings[0].shape),index = appliance\_readings[0].index,columns=appliance\_readings[0].columns)

`                    `for app\_reading in appliance\_readings:

`                        `train\_df+=app\_reading

`                `self.train\_mains.**append**(train\_df)

`                `for i,appliance\_name in **enumerate**(self.appliances):

`                    `self.train\_submeters[i].**append**(appliance\_readings[i])

`        `appliance\_readings = []

`        `for i,appliance\_name in **enumerate**(self.appliances):

`            `appliance\_readings.**append**((appliance\_name, self.train\_submeters[i]))

`        `self.train\_submeters = appliance\_readings   

`        `clf.partial\_fit(self.train\_mains,self.train\_submeters)



`    `def **test\_jointly**(self,d):

`        `*# store the test\_main readings for all buildings*

`        `for dataset in d:

`            `**print**("Loading data for ",dataset, " dataset")

`            `test=DataSet(d[dataset]['path'])

`            `for building in d[dataset]['buildings']:

`                `test.set\_window(start=d[dataset]['buildings'][building]['start\_time'],end=d[dataset]['buildings'][building]['end\_time'])

`                `test\_mains=**next**(test.buildings[building].elec.mains().load(physical\_quantity='power', ac\_type='apparent', sample\_period=self.sample\_period))

`                `if self.DROP\_ALL\_NANS and self.site\_only:

`                    `test\_mains, \_= self.**dropna**(test\_mains,[])

`                `if self.site\_only != True:

`                    `appliance\_readings=[]

`                    `for appliance in self.appliances:

`                        `test\_df=**next**((test.buildings[building].elec[appliance].load(physical\_quantity='power', ac\_type=self.power['appliance'], sample\_period=self.sample\_period)))

`                        `appliance\_readings.**append**(test\_df)



`                    `if self.DROP\_ALL\_NANS:

`                        `test\_mains , appliance\_readings = self.**dropna**(test\_mains,appliance\_readings)



`                    `if self.artificial\_aggregate:

`                        `**print** ("Creating an Artificial Aggregate")

`                        `test\_mains = **pd**.**DataFrame**(**np**.**zeros**(appliance\_readings[0].shape),index = appliance\_readings[0].index,columns=appliance\_readings[0].columns)

`                        `for app\_reading in appliance\_readings:

`                            `test\_mains+=app\_reading

`                    `for i, appliance\_name in **enumerate**(self.appliances):

`                        `self.test\_submeters.**append**((appliance\_name,[appliance\_readings[i]]))

`                `self.test\_mains = [test\_mains]

`                `self.storing\_key = **str**(dataset) + "\_" + **str**(building) 

`                `self.**call\_predict**(self.classifiers, test.metadata["timezone"])

`    `def **dropna**(self,mains\_df, appliance\_dfs=[]):

`        `"""

`        `Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection

`        `"""

`        `**print** ("Dropping missing values")

`        `*# The below steps are for making sure that data is consistent by doing intersection across appliances*

`        `mains\_df = mains\_df.dropna()

`        `ix = mains\_df.index

`        `mains\_df = mains\_df.loc[ix]

`        `for i in **range**(**len**(appliance\_dfs)):

`            `appliance\_dfs[i] = appliance\_dfs[i].dropna()



`        `for  app\_df in appliance\_dfs:

`            `ix = ix.intersection(app\_df.index)

`        `mains\_df = mains\_df.loc[ix]

`        `new\_appliances\_list = []

`        `for app\_df in appliance\_dfs:

`            `new\_appliances\_list.**append**(app\_df.loc[ix])

`        `return mains\_df,new\_appliances\_list





`    `def **store\_classifier\_instances**(self):

`        `"""

`        `This function is reponsible for initializing the models with the specified model parameters

`        `"""

`        `for name in self.methods:

`            `try:



`                `clf=self.methods[name]

`                `self.classifiers.**append**((name,clf))

`            `except **Exception** as e:

`                `**print** ("\n\nThe method {model\_name} specied does not exist. \n\n".**format**(model\_name=name))

`                `**print** (e)



`    `def **call\_predict**(self, classifiers, timezone):

`        `"""

`        `This functions computers the predictions on the self.test\_mains using all the trained models and then compares different learn't models using the metrics specified

`        `"""



`        `pred\_overall={}

`        `gt\_overall={}           

`        `for name,clf in classifiers:

`            `gt\_overall,pred\_overall[name]=self.**predict**(clf,self.test\_mains,self.test\_submeters, self.sample\_period, timezone)

`        `self.gt\_overall=gt\_overall

`        `self.pred\_overall=pred\_overall

`        `if self.site\_only != True:

`            `if gt\_overall.size==0:

`                `**print** ("No samples found in ground truth")

`                `return None

`            `for metric in self.metrics:

`                `try:

`                    `loss\_function = **globals**()[metric]                

`                `except:

`                    `**print** ("Loss function ",metric, " is not supported currently!")

`                    `continue

`                `computed\_metric={}

`                `for clf\_name,clf in classifiers:

`                    `computed\_metric[clf\_name] = self.**compute\_loss**(gt\_overall, pred\_overall[clf\_name], loss\_function)

`                `computed\_metric = **pd**.**DataFrame**(computed\_metric)

`                `**print**("............ " ,metric," ..............")

`                `**print**(computed\_metric) 

`                `self.errors.**append**(computed\_metric)

`                `self.errors\_keys.**append**(self.storing\_key + "\_" + metric)

`        `if self.display\_predictions:

`            `if self.site\_only != True:

`                `for i in gt\_overall.columns:

`                    `**plt**.**figure**()

`                    `*#plt.plot(self.test\_mains[0],label='Mains reading')*

`                    `**plt**.**plot**(gt\_overall[i],label='Truth')

`                    `for clf in pred\_overall:                

`                        `**plt**.**plot**(pred\_overall[clf][i],label=clf)

`                        `**plt**.**xticks**(rotation=90)

`                    `**plt**.**title**(i)

`                    `**plt**.**legend**()

`                `**plt**.**show**()



`    `def **predict**(self, clf, test\_elec, test\_submeters, sample\_period, timezone ):

`        `**print** ("Generating predictions for :",clf.MODEL\_NAME)        

`        `"""

`        `Generates predictions on the test dataset using the specified classifier.

`        `"""



`        `*# "ac\_type" varies according to the dataset used.* 

`        `*# Make sure to use the correct ac\_type before using the default parameters in this code.*   





`        `pred\_list = clf.disaggregate\_chunk(test\_elec)

`        `*# It might not have time stamps sometimes due to neural nets*

`        `*# It has the readings for all the appliances*

`        `concat\_pred\_df = **pd**.**concat**(pred\_list,axis=0)

`        `gt = {}

`        `for meter,data in test\_submeters:

`                `concatenated\_df\_app = **pd**.**concat**(data,axis=1)

`                `index = concatenated\_df\_app.index

`                `gt[meter] = **pd**.**Series**(concatenated\_df\_app.values.**flatten**(),index=index)

`        `gt\_overall = **pd**.**DataFrame**(gt, dtype='float32')

`        `pred = {}

`        `if self.site\_only ==True:

`            `for app\_name in concat\_pred\_df.columns:

`                `app\_series\_values = concat\_pred\_df[app\_name].values.flatten()

`                `pred[app\_name] = **pd**.**Series**(app\_series\_values)

`            `pred\_overall = **pd**.**DataFrame**(pred,dtype='float32')

`            `pred\_overall.plot(label="Pred")

`            `**plt**.**title**('Disaggregated Data')

`            `**plt**.**legend**()

`        `else:

`            `for app\_name in concat\_pred\_df.columns:

`                `app\_series\_values = concat\_pred\_df[app\_name].values.flatten()

`                `*# Neural nets do extra padding sometimes, to fit, so get rid of extra predictions*

`                `app\_series\_values = app\_series\_values[:**len**(gt\_overall[app\_name])]

`                `pred[app\_name] = **pd**.**Series**(app\_series\_values, index = gt\_overall.index)

`            `pred\_overall = **pd**.**DataFrame**(pred,dtype='float32')



`        `return gt\_overall, pred\_overall

`    `*# metrics*

`    `def **compute\_loss**(self,gt,clf\_pred, loss\_function):

`        `error = {}

`        `for app\_name in gt.columns:

`            `error[app\_name] = loss\_function(gt[app\_name],clf\_pred[app\_name])

`        `return **pd**.**Series**(error)   






