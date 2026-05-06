El siguiente es un ejemplo de una tesis llamada "Implementación de una aplicación web de soporte al proceso de 
evaluación de comunicabilidad". Est servirá de guia para la redaccion del documento de las diferentes secciones.

---

**Problema seleccionado:** 
Por lo expuesto anteriormente, el presente proyecto de tesis busca abordar el 
principal problema identificado; es decir, las dificultades que existen al llevar a cabo el 
Método de Evaluación de Comunicabilidad de manera manual para evaluar aplicaciones 
de software.  

**1.2 Objetivos** 
Objetivo general 
Implementar una aplicación web que dé soporte a la ejecución del Método de 
Evaluación de Comunicabilidad (CEM) para evaluar aplicaciones de software. 

**Objetivos específicos**  
- O 1. Modelar el Método de Evaluación de Comunicabilidad (CEM) con base en la 
literatura y criterios de especialistas detallando las actividades requeridas en cada 
una de sus etapas. 
- O 2. Implementar un módulo de soporte a las etapas de planificación y seguimiento 
de las etapas del método CEM de carácter colaborativo que permita aplicar 
buenas prácticas relacionadas a la gestión de información. 
- O 3. Implementar un módulo de soporte a la captura y procesamiento de datos en las 
etapas de análisis e información del método CEM. 
7 
- O 4. Validar el aplicativo web desarrollado a través de un caso de estudio en el que se 
demuestre su utilidad y contribución a la ejecución del método de evaluación de 
comunicación (CEM) como herramienta de soporte. 

**Resultados esperados** 

O 1. Modelar el Método de Evaluación de Comunicabilidad (CEM) en base a la 
literatura y criterios de especialistas detallando las actividades requeridas en cada 
una de sus etapas. 
- R 1.1. Informe sobre entrevista realizada a especialistas en Interacción Humano
Computador (HCI) sobre la forma de aplicación del método CEM.  
- R 1.2. Cuadro comparativo de las diferentes actividades realizadas en la 
literatura para llevar a cabo el método CEM. 
- R 1.3. Modelado del proceso para llevar a cabo el método CEM. 

O 2. Implementar un módulo de soporte a las etapas de planificación y seguimiento 
de las etapas del método CEM de carácter colaborativo que permita aplicar 
buenas prácticas relacionadas a la gestión de información. 
- R 2.1. Lista de requisitos del módulo de planificación y seguimiento. 
- R 2.2. Prototipo del módulo de planificación y seguimiento. 
- R 2.3. Arquitectura del sistema para el módulo de planificación y seguimiento. 
- R 2.4. Desarrollo del módulo de planificación y seguimiento. 
- R 2.5. Informe de los resultados de las pruebas de aceptación aplicadas al 
módulo de planificación y seguimiento. 

O 3. Implementar un módulo de soporte a la captura y procesamiento de datos en las 
etapas de análisis e información del método CEM. 
- R 3.1. Lista de requisitos del módulo de captura y procesamiento de datos. 
- R 3.2. Prototipo del módulo de captura y procesamiento de datos. 
- R 3.3. Arquitectura actualizada del sistema para el módulo de captura y 
procesamiento de datos. 
- R 3.4. Desarrollo del módulo de captura y procesamiento de datos. 
- R 3.5. Informe de los resultados de las pruebas de aceptación aplicadas al 
módulo de captura y procesamiento de datos. 

O 4. Validar el aplicativo web desarrollado a través de un caso de estudio en el que se 
demuestre su utilidad y contribución a la ejecución del método de evaluación de 
comunicación (CEM) como herramienta de soporte. 
- R 4.1. Planificación y diseño del caso de estudio. 
- R 4.2. Análisis de los resultados obtenidos de la ejecución del caso de estudio.

---

**Capitulo 4. Modelado del Método de Evaluación de Comunicabilidad (CEM) sobre la base de la literatura y criterios de especialistas**

**4.1 Introducción**
Este capítulo tiene como finalidad presentar los resultados obtenidos para el primer 
objetivo planteado, el cual, corresponde al modelado del Método de Evaluación de 
Comunicabilidad (CEM). El propósito de este objetivo es de uniformizar y establecer de 
forma detallada la lista de actividades necesarias para la ejecución del método debido a la 
existencia de múltiples criterios y perspectivas de diferentes especialistas para aplicarlo. 
Con el fin de llevar a cabo una estandarización del proceso, se realizó una revisión 
sistemática de la literatura, como también se llevaron a cabo dos entrevistas 
semiestructuradas con dos especialistas en el área de HCI que posean experiencia en la 
aplicación de este método. En primer lugar, a partir de la revisión sistemática de la 
literatura, se pretende realizar un cuadro comparativo con las diferentes actividades que 
se mencionan en la literatura al aplicarse este método, con ello se puede comparar y 
evidenciar las actividades más recurrentes para el modelado del proceso. Y, por otro lado, 
a partir de la opinión de expertos en el área de HCI, se obtiene información más cercana 
respecto a la forma de aplicación de este método en la práctica.

Tras recopilar esta información y visualizarla en un cuadro comparativo, se ha 
realizado un diagrama en notación BPMN 2.0 sobre la aplicación del método CEM 
considerando las cinco fases que propone su autora de Souza (de Souza y Faria, 2009). A 
partir de esto, se pretende formalizar y evidenciar los pasos necesarios para llevar a cabo 
el método de manera detallada con la finalidad de que se detecten correctamente los 
problemas de comunicabilidad y con ello generar información más precisa que permita a 
los investigadores generar interfaces más entendibles. 
 
A continuación, se presentan los resultados alcanzados que permitieron lograr el 
cumplimiento del objetivo a partir de sus descripciones, medios de verificación y el 
cumplimiento de los indicadores objetivamente verificables. 

**4.2 Resultados Alcanzados** 
**Informe sobre entrevista realizada a especialistas en Interacción Humano Computador (HCI) sobre la forma de aplicación del método CEM** 

Para el primer resultado alcanzado, se llevaron a cabo entrevistas con dos 
especialistas en el área de Interacción Humano Computador con el propósito de recopilar 
información acerca de las actividades que estos suelen realizar al aplicar el Método de 
Evaluación de Comunicabilidad. A partir de ello, se generó un informe con la información 
sintetizada y resumida; además, se utilizó el instrumento de cuadro comparativo para 
mostrar los pasos que los especialistas indicaron aplicar cuando han realizado 
evaluaciones de comunicabilidad. Tras llevarse a cabo este primer resultado, el informe 
ha sido utilizado como insumo para realizar el modelado del proceso en un diagrama 
BPMN del siguiente resultado esperado. 
En primer lugar, se llevaron a cabo entrevistas semiestructuradas, por ello, se 
generó un guion de preguntas que serviría como base para obtener la información 
pertinente. Dicho guion se encuentra en el Anexo I.  
Una vez realizada la estructura de las preguntas, se llevaron a cabo las entrevistas, 
las cuales habían sido previamente coordinadas con los especialistas y aprobadas a partir 
de la firma de un consentimiento informado, el cual se encuentra en el Anexo J. Estas 
entrevistas tenían la característica de ser de manera virtual, por lo cual se utilizó la 
plataforma de videoconferencias Zoom, la cual permitió grabar las sesiones de preguntas.  

Tras realizar las preguntas estructuradas a los especialistas, como medio de 
verificación de resultado, se procedió a realizar un informe, desarrollado en el Anexo K, 
de las entrevistas a partir de la revisión de las grabaciones hechas. A partir del informe, 
se construyó una tabla plasmando las respuestas a cada una de las preguntas planteadas 
respecto a las actividades realizadas por cada especialista. Por tal motivo en el Anexo L, 
se muestra un cuadro comparativo con las diferentes actividades que los especialistas han 
llevado a cabo en las distintas fases del Método de Evaluación de Comunicabilidad.  
A partir del cuadro comparativo, se evidencian las diferentes formas de aplicación 
del método CEM debido a la diferente visión sobre su uso al momento de evaluar 
interfaces. 
Finalmente, como indicador objetivamente verificable, se ha elaborado un informe 
donde se evidencia que los evaluadores han contestado al 100% de las preguntas 
planteadas en el guion construido, el cual se encuentra en el Anexo K. 
Cuadro comparativo de las diferentes actividades realizadas en la literatura 
para llevar a cabo el método CEM 
A partir del segundo resultado esperado, se recopiló la información acerca de las 
actividades reportadas en la literatura para evaluar la comunicabilidad de interfaces a 
partir de las cinco fases que establece el método de CEM. Los estudios analizados han 
sido obtenidos a partir de la revisión sistemática realizada previamente en el Capítulo 3, 
en donde, a partir de la revisión del estado del arte se busca responder la pregunta de 
investigación N°2 “¿Existen metodologías, métodos, protocolos o procesos de trabajo 
para llevar a cabo el método de evaluación de comunicabilidad y cómo se aplican?”. Tras 
seleccionar las fuentes, se utilizó el instrumento de cuadro comparativo para evidenciar 
las actividades que cada uno de los estudios seleccionados aplicó en las cinco fases para 

llevar a cabo la evaluación de comunicabilidad. Por tal motivo, como medio de 
verificación de resultado, se observa en el Anexo M, el cuadro comparativo con el número 
y porcentaje de actividades reportadas por los 19 estudios escogidos previamente. 
A partir de la construcción de la tabla mostrada en Anexo N, se visualiza las fases 
en donde una cantidad de artículos académicos reporta con mayor frecuencia la aplicación 
de ciertas actividades. Mientras que, por otro lado, se observa que, en las fases de 
interpretación y elaboración de perfil semiótico, algunos artículos no reportan la ejecución 
de actividades esenciales de dichas fases.  
Finalmente, el indicador objetivamente verificable de este resultado se visualiza 
en el Anexo E, en donde se han considerado el 100% de los artículos relevantes para la 
elaboración del cuadro comparativo.

**4.2.3 Modelado del proceso para llevar a cabo el método CEM**
[Contenido]

**4.3 Discusión**
Se llevaron a cabo tres resultados esperados para lograr el objetivo de modelar el 
Método de Evaluación de Comunicabilidad (CEM). Para el primer resultado, se desarrolló 
un informe detallado sobre la aplicación del método CEM mediante entrevistas a dos 
especialistas en el área de Interacción Humano Computador (HCI). Para el segundo 
resultado esperado, se desarrolló un cuadro comparativo de las actividades realizadas en 
la literatura para llevar a cabo el método CEM, donde se destaca que no todos los 
evaluadores realizan todas las tareas para las fases fundamentales que establece el método 
y en algunos casos, se llega a omitir la fase de construcción de perfil semiótico, la cual 
permite identificar las características del usuario que utiliza la interfaz web. Finalmente, 
para el tercer resultado esperado, a partir de la información recopilada en las entrevistas 
y literatura, se llevó a cabo el modelado del proceso para llevar a cabo el método CEM, 
el cual fue validado por especialistas en el área de HCI, los cuales propusieron algunas 
mejoras respecto a la notación BPMN 2.0 utilizada para el modelado.  
De manera adicional, el desarrollo de la discusión para cada uno de los resultados 
esperados se encuentra en el Anexo Q. 

---

Anexo Y : Prototipo de alto nivel del módulo de planificación y seguimiento del 
Método de Evaluación de Comunicabilidad

    Prototipo de alto nivel del módulo de planificación y seguimiento del Método de 
    Evaluación de Comunicabilidad[LINK enlazado al nombre]

Anexo BB : Diagrama de casos de uso del módulo de planificación y seguimiento 
del Método de Evaluación de Comunicabilidad 

    [Capturas de pantalla de su diagrama de fases]

Anexo LL : Documento del catálogo de pruebas de aceptación para el módulo de 
planificación y seguimiento del Método de Evaluación de Comunicabilidad 

    Documento de catálogo de pruebas de aceptación del módulo de planificación y 
    seguimiento[LINK enlazado al nombre] 