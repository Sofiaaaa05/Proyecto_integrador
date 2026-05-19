# Proyecto_integrador
# Histórico de ventas de un restaurante para predicción de demanda mediante IA
# Integrantes
Carla Karina Rocha Hernandez

Lidia Sofia Hernandez Briseño

Maria Gabriela Salas Romo

Santiago Olivares Legaspi


# Introduccion
Realizamos 4 tipos de modelos (Tree Regressor, Random Forest, Gradient Boosting y XGBoosting) para en nuestro caso saber la prediccion de ventas de un restaurante de mariscos, nuestra base de datos fue manual, ya que originakmente la base fue sacada del programa Soft Restaurant 11, importada, limpiada y manipulada para esta prediccion de datos. Nos enfocamos en usar una clave para cada venta, una descripcion general de cada producto, sus ventas en el dia y las fechas de estas.

# Planteamineto del problema 
El problema que queremos reducir y si es posible mitigar seria que el negocio generara perdida, tanto economicas comon desperdicio de producto. Ya que al no llevar una organizacion de los productos que mas o menos se venden, estos productos menos vendidos son desperdiciados sin generar ganancia y en los productos mas vendidos al no conocer que este producto atrae mas al cliente nos quedamoos sin cantidad suficinte de este producto lo cual lleva a no generar ganacias y solo quedarnos con los productos que menos interaccion tiene con los clientes.

# Descripcion del dataset
Nuestro dataset cuenta con 31377 filas y 11 columnas, sumando un total de 345147 datos. Los cuales se dividen en secciones empezando por una clave unica para cada producto, su descripcion y el grupo al que pertenece ya sea tipo de bebida o tipo de comida, el precio unitarios de cada articulo, la cantidad vendida y su total de venta determinado dia; ya que los datos tambien los dividimos por dia, mes, año y dia de la semana.

# Desarrollo teorico
Antes de cualquier calculo primero que nada todo este procedimiento fue realizado en google colab dodne para sascar el resultado de nuestros modelos tuvimos que dar especificaiones y ajustes a la base de datos, hicimos el siguiente procedimiento

# 1.Ubicamos nuestra variable objetivo para estabilizar la varianza
  y = np.log1p(df['VENTA_TOTAL_DIA'])
  
# 2.Procesamos el set de entrenamiento 
train_temporal = X_train.copy()
train_temporal['VENTA_LOG'] = y_train

# 3.Calcular el promedio de cada producto con train
producto_map_log = train_temporal.groupby('DESCRIPCION')['VENTA_LOG'].mean()

# 4.Aplicamos mapeo a cada conjunto 
X_train['PRODUCTO_ENCODED'] = X_train['DESCRIPCION'].map(producto_map_log)
X_test['PRODUCTO_ENCODED'] = X_test['DESCRIPCION'].map(producto_map_log)

si tenemos un test sin un train visto se asigna el promedio general
X_test['PRODUCTO_ENCODED'] = X_test['PRODUCTO_ENCODED'].fillna(promedio_global_log)

# 5.limpoeza final 
X_train = X_train.drop(columns=['DESCRIPCION'])
  X_test = X_test.drop(columns=['DESCRIPCION'])

Nosotros utilizamos las siguientes formulas para extraer los resultados de los siguientes modelos de regresion que implementamos en nuestro proyecto:

# Tree Regressor:
print("Entrenando Decision Tree...")
tree_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_real_tree = np.expm1(tree_reg.predict(X_test))
R2=0.7018
RMSE=$616.29 

# Random Forest:
print("Entrenando Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=None, oob_score=True, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos los núcleos para ir más rápido
rf.fit(X_train, y_train)
y_pred_real_rf = np.expm1(rf.predict(X_test))
R2=0.7574
RMSE=$555.87

# Gradient Boosting:
print("Entrenando Gradient Boosting...")
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred_real_gbr = np.expm1(gbr.predict(X_test))
R2=0.7238
RMSE=$593.16

# XGBoosting:
print("Entrenando XGBoost...")
xgb_reg = xgb.XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred_real_xgb = np.expm1(xgb_reg.predict(X_test))
R2=0.7659
RMSE=$546.06

Los R2 y RMSE los sacamos por la siguiente formula:

for nombre, preds in modelos_dict.items():
    rmse_r = np.sqrt(mean_squared_error(y_test_real, preds))
    r2_r = r2_score(y_test_real, preds)
    print(f"{nombre:<30} | {r2_r:<10.4f} | ${rmse_r:<11.2f}")
    
# Aplicacion practica del modelo 
De esta manera es que sirve cada uno de los siguientes modelos en nuestro producto si lo aplicaramos en la vida real:

Tree Regressor: Es un modelo que se basa en preguntas si o no, por lo que no es tan exacto y en este caso ignorar datos como la fecha, dia, año; por lo que si un producto vende mucho, el árbol va a predecir buenas ventas sea lunes o sabado, de hecho como nos podemos dar cuenta su error es el mas alto con 616.29

Random Forest: Es un modelo mas democratico, ya que se convina con el voto de los árboles especialstaa en productos y fechas, ya no es un árbol de decisión tan simple, teniendo un error mas bajo que el anterior de 555.87

Gradient Boosting: Aqui al enfocarse en arboles de secuencia, o sea, uno trabaja el siguiente corrige y tenerlo en una profundidad de 3 limmita la variedad de productos del dataset quedando en un erro de 593.16. Sin embargo asumiendo que subimos la profundidad aunque bajaria significativamente su error, si lo comparamos con xgbooosting sigue siendo mejor; ya que para empezar aunque tendria mas exactitudes estas pueden ser tan pequeñas que al minimo cambio de ventas en determinado dia de otro año se descontrolarian y cairiamos en un sobreajuste, tambien este modelo es mucho mas tardado al ser uno por uno.

XGBoosting: Al tener una profundidad de 10 y corregir errores en cadena el modelo encuentra un equilibrio dodne sa prioridad al producto, pero tomando en cuenta el año y dia; por lo que toma hasta el ultimo dato es fechas sin perder el foco central en los productos. obteniendo un margen de error mas bajo en dinero real 546.06 y obtiene un R2 más alto.

# Discusion 
Al poner a prueba todos estos modelos relacionados entre si, pero con diferencias significativas descubrimos que al final nuestro mejor modelo con un margen de error bajo de 546.06 es el XGBossting esto se debe al metodo que utiliza este modelo lo cual lo hace mas eficiente, ya que no solo se enfoca en el producto y sus ventas, se enfoca en los otros factores que tenemos en nuestra base, como la fecha, dia, mes, año y dia de la semana.  Sin embargo el modelo se limita al no tener elementos externos como el clima o eventos festivos que justifican ese error.

# Conclusiones

# Referencias 
