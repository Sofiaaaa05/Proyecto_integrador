# Proyecto_integrador
# Histórico de ventas de un restaurante para predicción de demanda mediante IA
# Integrantes
Carla Karina Rocha Hernandez

Lidia Sofia Hernandez Briseño

Maria Gabriela Salas Romo

Santiago Olivares Legaspi


# Introduccion
Desarrollamos cuatro modelos de regresión (Decision Tree, Random Forest, Gradient Boosting y XGBoost) con el objetivo de predecir las ventas de un restaurante de mariscos. La información base se extrajo del sistema Soft Restaurant 11 en formato .bak, el cual fue restaurado y analizado en SQL. Debido a la complejidad del software y su estructura de múltiples tablas relacionales, aplicamos técnicas de ingeniería de datos mediante consultas, agregaciones y uniones (JOINs) para consolidar la información. Este procesamiento nos permitió estructurar un histórico de patrones de venta esencial para estimar la demanda futura y controlar las mermas. Finalmente, para el entrenamiento de los modelos, nos enfocamos en cuatro variables principales: clave única de venta, descripción general del producto, volumen de ventas del día y fecha de registro.

# Planteamineto del problema 
El objetivo de este proyecto es reducir y, de ser posible, mitigar las pérdidas económicas y el desperdicio de mercancía en el negocio. Actualmente, la falta de control sobre la rotación de inventario genera dos problemas críticos: por un lado, los productos de baja demanda se vencen o desperdician sin generar ingresos; por otro lado, la falta de stock de los productos más vendidos provoca la pérdida de oportunidades de venta, dejando al negocio con un inventario estancado y de bajo interés para los clientes.

# Descripcion del dataset
El conjunto de datos bajo análisis está compuesto por 31,377 registros (filas) y 11 variables (columnas), consolidando un total de 345,147 observaciones. Estructuralmente, el dataset se compone de variables categóricas; tales como el identificador único del producto, su descripción general y la categoría de consumo (comida o bebida), y variables numéricas continuas que detallan el aspecto comercial, incluyendo el precio unitario, la cantidad de unidades vendidas y los ingresos totales percibidos en un día específico. Asimismo, se integraron variables de corte temporal desglosadas por día, mes, año y día de la semana. Tras una evaluación de la calidad de la información, se constató la ausencia de valores nulos o registros atípicos (outliers); por su parte, la presencia de registros con características idénticas en productos, precios o cantidades responde de forma natural al comportamiento histórico y repetitivo de las transacciones diarias del negocio.

# Desarrollo teorico
Antes de cualquier calculo primero que nada todo este procedimiento fue realizado en google colab dodne para sascar el resultado de nuestros modelos tuvimos que dar especificaiones y ajustes a la base de datos, hicimos el siguiente procedimiento

<img width="737" height="336" alt="image" src="https://github.com/user-attachments/assets/fa77dafa-eb0e-46f9-a3d6-f143a5e45b85" />
<img width="1268" height="464" alt="image" src="https://github.com/user-attachments/assets/94ce2dd1-043c-47e0-adce-e2b343dbe324" />
<img width="1277" height="514" alt="image" src="https://github.com/user-attachments/assets/411f2698-4ffa-4564-b040-d833c89f6b8a" />
<img width="798" height="625" alt="image" src="https://github.com/user-attachments/assets/2287f7de-cee7-423a-9cde-baa5b9163fef" />
<img width="815" height="599" alt="image" src="https://github.com/user-attachments/assets/ac1071f5-5a17-4786-8e2e-26cb1404a775" />
<img width="1452" height="495" alt="image" src="https://github.com/user-attachments/assets/a80a92df-99e8-4c61-998e-611ea1652f98" />
<img width="713" height="490" alt="image" src="https://github.com/user-attachments/assets/0bfa70b2-426e-4d21-afe5-220424f2f9de" />
<img width="676" height="562" alt="image" src="https://github.com/user-attachments/assets/238e2abf-aa82-40f8-91bb-31922a740850" />
<img width="751" height="590" alt="image" src="https://github.com/user-attachments/assets/c36dd0bd-d831-4d22-bc1c-5eff4a0c1bb2" />
<img width="875" height="563" alt="image" src="https://github.com/user-attachments/assets/025cf344-5107-4e14-a017-bc1870b1dfb9" />
<img width="1733" height="665" alt="image" src="https://github.com/user-attachments/assets/e1c491b2-aa04-4dfe-b921-188c9d23f471" />
<img width="780" height="728" alt="image" src="https://github.com/user-attachments/assets/45ad1191-427b-4a56-b16e-282abe623f8a" />
<img width="745" height="264" alt="image" src="https://github.com/user-attachments/assets/5da4ca7b-39ef-4e69-b558-2f45b194acb5" />
<img width="1733" height="614" alt="image" src="https://github.com/user-attachments/assets/c51b2cad-ad03-4ac6-8abf-f0ebee050782" />
<img width="1723" height="711" alt="image" src="https://github.com/user-attachments/assets/4b499100-9d06-4737-819c-2b974cef0392" />
<img width="1758" height="654" alt="image" src="https://github.com/user-attachments/assets/3248b7c9-0085-4e62-a89d-3cd92ac4a8dd" />
<img width="1770" height="501" alt="image" src="https://github.com/user-attachments/assets/20dcdaae-4981-4084-b53c-910ec002bb23" />
<img width="753" height="342" alt="image" src="https://github.com/user-attachments/assets/570e2000-ba23-4786-9df3-3447b40cc56a" />
<img width="871" height="669" alt="image" src="https://github.com/user-attachments/assets/3cc0fa6a-03d6-4969-adfb-ed378aba68b1" />
<img width="642" height="649" alt="image" src="https://github.com/user-attachments/assets/ff6eb3d5-3ee2-4fde-9f80-3ab06a6dd14d" />
<img width="677" height="579" alt="image" src="https://github.com/user-attachments/assets/6114f1e1-046e-455d-98ec-40d94e2c5d4a" />
<img width="929" height="520" alt="image" src="https://github.com/user-attachments/assets/f6e4cbd5-3526-4e7e-9f19-71b10e4ea37b" />
<img width="987" height="520" alt="image" src="https://github.com/user-attachments/assets/f0fb0a11-88b4-44ee-8f9f-9c073eea8f40" />
<img width="1123" height="748" alt="image" src="https://github.com/user-attachments/assets/ee0a6804-f06a-41d6-a731-2143d7b397b2" />

# Aplicacion practica del modelo 
De esta manera es que sirve cada uno de los siguientes modelos en nuestro producto si lo aplicaramos en la vida real:

Tree Regressor: Es un modelo que se basa en preguntas si o no, por lo que no es tan exacto y en este caso ignora datos como la fecha, dia, año; por lo que si un producto vende mucho, el árbol va a predecir buenas ventas sea lunes o sabado, de hecho como nos podemos dar cuenta su error es el mas alto con 616.29

Random Forest: Es un modelo mas democratico, ya que se convina con el voto de los árboles especialistas en productos y fechas, ya no es un árbol de decisión tan simple, teniendo un error mas bajo que el anterior de 555.87

Gradient Boosting: Aqui al enfocarse en arboles de secuencia, o sea, uno trabaja, el siguiente corrige y tenerlo en una profundidad de 3 limmita la variedad de productos del dataset quedando en un erro de 593.16. Sin embargo asumiendo que subimos la profundidad aunque bajaria significativamente su error, si lo comparamos con xgboos sigue siendo mejor; ya que para empezar aunque tendria mas exactitudes estas pueden ser tan pequeñas que al minimo cambio de ventas en determinado dia de otro año se descontrolarian y cairiamos en un sobreajuste, tambien este modelo es mucho mas tardado al ser uno por uno.

XGBoost: Al tener una profundidad de 10 y corregir errores en cadena el modelo encuentra un equilibrio donde da prioridad al producto, pero tomando en cuenta el año y dia; por lo que toma hasta el ultimo dato es fechas sin perder el foco central en los productos. obteniendo un margen de error mas bajo en dinero real 546.06 y obtiene un R2 más alto.

# Discusion 
Al poner a prueba todos estos modelos relacionados entre si, pero con diferencias significativas descubrimos que al final nuestro mejor modelo con un margen de error bajo de 546.06 es el XGBoost esto se debe al metodo que utiliza este modelo lo cual lo hace mas eficiente, ya que no solo se enfoca en el producto y sus ventas, se enfoca en los otros factores que tenemos en nuestra base, como la fecha, dia, mes, año y dia de la semana.  Sin embargo el modelo se limita al no tener elementos externos como el clima o eventos festivos que justifican ese error.

# Conclusiones
El presente proyecto demostró la viabilidad y el impacto de implementar modelos de aprendizaje supervisado basados en árboles de decisión para optimizar la cadena de suministro y la gestión de inventarios en un restaurante de mariscos. A través de la predicción de la demanda, se logró dar una respuesta directa a la problemática inicial: mitigar las pérdidas económicas por desabasto de productos de alta rotación y reducir el desperdicio (mermas) de artículos con baja demanda. Tambien se demostro que el XGBoost permitirá al restaurante anticipar el volumen de ventas diarias por producto. Esto se traduce en una ventaja operativa clave: planificar las compras de materia prima fresca con base en datos históricos y no en intuiciones.

# Referencias 
An Introduction to Statistical Learning. (s. f.). An Introduction To Statistical Learning. https://www.statlearning.com/
VanderPlas, J. (s. f.). Python Data Science Handbook | Python Data Science Handbook. https://jakevdp.github.io/PythonDataScienceHandbook/
