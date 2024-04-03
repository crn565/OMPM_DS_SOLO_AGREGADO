## OMPM_DS_SOLO_AGREGADO ##

# Desagregación del consumo energetico  partiendo  de un modelo ya entrenado contra un Dataset que contiene sólo el Agregado #

En este  proyecto  se  genera un modelo  partiendo de datos de medidas del OMPM. Posteriormente  usando ese modelo ya entrenado (cuadernos 1 al 7) , partiendo de este  se trata de desagregar la demanda partiendo unicamente de un dataset qeu solo contiene las medidas del agregado 
El ultimo cuaderno, que es el mas interesante muestra el uso de siteonlyapi, una nueva interfaz de NILMTK que es una modificación de ExperimentAPI de NILMTK y que permite a los usuarios de NILMTK obtener las demandas de energía de sus hogares/edificios para diferentes electrodomésticos potenciales partiendo de un conjunto de datos ya entrenado.


Comencemos con los datos deL OMPM  para demostrar el uso de esta API. Este experimento muestra cómo el usuario puede convertir los datos de su medidor en el formato DSUAL adecuado y llamar a la API para desagregar la energía en demandas de electrodomésticos en función del conjunto de entrenamiento. Se convierten los datos del medidor al formato adecuado. Asimismo cambiamos las fechas de inicio y finalización de su conjunto de datos de prueba y también, ingresamos los valores de los diferentes parámetros en el diccionario. Dado que necesitamos varios electrodomésticos, ingresamos los nombres de todos los electrodomésticos requeridos en el parámetro 'appliances'. También mencionamos site_only es cierto porque queremos desagregar los datos del medidor del sitio solo sin ninguna comparación con los datos del submedidor.

Es crucial pues  como se tipifica los aplicativos,los metodos y los conjuntos de datos,  el cual a continuacion  podemos  ver a continuación:


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

## Explicacion del código ##

El código proporcionado inicializa un objeto de **API** mediante la configuración del experimento proporcionada. A continuación, se muestra un desglose de la configuración del experimento

- **Especificación de alimentación:** La alimentación de la red y del aparato se especifica como potencia "activa".
- **Frecuencia de muestreo:** La frecuencia de muestreo se establece en 60 segundos.
- **Dispositivos:**  El experimento incluye dispositivos con identificadores 2, 3, 4, 5 y 6. 
- **Métodos/Algoritmos:** Se utilizan cuatro algoritmos NILM: CO, FHMM, Mean y Hart. 
- **Modo solo de sitio:** Este modo está habilitado (site_only=True), lo que significa que solo se tiene en cuenta la potencia agregada a nivel de sitio para el entrenamiento y las pruebas. 
- **Conjunto de datos de entrenamiento:**  El conjunto de datos de entrenamiento procede del conjunto de datos 'DSUAL', almacenado en el archivo 'ualm2.h5'. Se utilizan datos del edificio 1, a partir de '2023-02-24 14:47:10' y hasta '2023-02-24 20:03:54'. 
- **Conjunto de datos de prueba:**  El conjunto de datos de prueba proviene del conjunto de datos 'DSUAL', almacenado en el archivo 'ualm3.h5'. Se utilizan los datos del edificio 1, a partir de '2023-05-13 17:22:02' y hasta '2023-05-13 23:43:15'. 
- **Métrica de evaluación:** Se elige RMSE (error cuadrático medio) como métrica de evaluación.


El objeto de API instanciado con esta configuración llevará a cabo el experimento NILM utilizando los algoritmos y conjuntos de datos especificados, y evaluará su rendimiento mediante RMSE


Cuando se llama api_results_experiment_1 = API(experiment1) con el contenido de experiment1, se llaman las siguientes funciones de la clase API:
1.	**Constructor (__init__):**
•	Se inicializan los atributos de la clase API con los parámetros proporcionados en experiment1.
•	Se llama al método experiment para iniciar el experimento.
2.	**Método experiment:**
•	Este método es llamado por el constructor.
•	Se iteran los clasificadores (métodos/algoritmos) especificados en experiment1['methods'].
•	Dependiendo de si la opción de entrenamiento por fragmentos está habilitada o no, se llaman a los métodos train_chunk_wise o train_jointly para entrenar los clasificadores.
•	Después del entrenamiento, se realizan pruebas ya sea por fragmentos o conjuntamente llamando a los métodos test_chunk_wise o test_jointly.
3.	**Métodos de Entrenamiento y Prueba:**
•	Se llaman los métodos de entrenamiento train_chunk_wise o train_jointly dependiendo de si el entrenamiento se realiza por fragmentos o conjuntamente.
•	Se llaman los métodos de prueba test_chunk_wise o test_jointly dependiendo de si las pruebas se realizan por fragmentos o conjuntamente.
4.	**Otros Métodos Auxiliares:**
•	Los métodos auxiliares como dropna, store_classifier_instances, call_predict, predict, y compute_loss pueden ser llamados indirectamente dentro de otros métodos mencionados anteriormente, dependiendo de la lógica de la implementación y el flujo de ejecución.
En resumen, cuando se llama **API(experiment1)**, se ejecutan una serie de métodos dentro de la clase **API** para realizar el experimento de desagregación de energía





## CONTENIDO DE LA API ##

Esta es una clase de Python llamada API diseñada para experimentos de NILM (Monitoreo de Cargas No Intrusivo). Te permite entrenar y probar varios algoritmos de desagregación en un conjunto de datos de lecturas de electricidad de edificios.

Aquí tienes un desglose de la clase:


**Initialization:**

- \_\_init\_\_(self, params): Esta función inicializa la clase con parámetros como electrodomésticos, conjuntos de datos, métodos de entrenamiento, etc.

**Experiment Functions:**

- experiment(self): Esta función llama a las funcionalidades de entrenamiento y prueba.. 
  - train\_chunk\_wise(self, clf, d, current\_epoch): Esta función entrena los clasificadores de manera fragmentada para una mejor gestión de la memoria..
  - test\_chunk\_wise(self, d): Esta función prueba los clasificadores de manera fragmentada..
  - train\_jointly(self, clf, d): .Esta función entrena los clasificadores en todo el conjunto de datos de manera conjunta.
  - test\_jointly(self, d): Esta función prueba los clasificadores en todo el conjunto de datos de manera conjunta.
**Data Handling Functions:**

- dropna(self, mains\_df, appliance\_dfs=[]): Esta función elimina las filas con valores faltantes de los marcos de datos de la corriente principal y de los electrodomésticos, manteniendo la consistencia.
**Model Handling Functions:**

- store\_classifier\_instances(self): Esta función inicializa los modelos basados en los métodos especificados.
**Prediction and Evaluation Functions:**

- call\_predict(self, classifiers, timezone): Esta función calcula predicciones en los datos de prueba utilizando todos los modelos y los compara utilizando métricas especificadas.
- predict(self, clf, test\_elec, test\_submeters, sample\_period, timezone ): Esta función genera predicciones para un clasificador específico en el conjunto de datos de prueba.
- compute\_loss(self,gt,clf\_pred, loss\_function): Esta función calcula la pérdida (error) entre la verdad fundamental y las predicciones para cada electrodoméstico.

**Other Functions:**

- display\_predictions(self): Esta función traza la verdad fundamental y el consumo de energía predicho para cada electrodoméstico.






## Código Python ##


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






