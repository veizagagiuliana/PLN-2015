#Practico 2

##Ejercicio 1: Corpus Ancora: Estadísticas de etiquetas POS

oraciones: 17379
ocurrencia de palabras: 517300
vocabulario de palabras: 46483
vocabulario de etiquetas: 48


|ETIQUETAS |   SIGNIFICADO           | CANTIDAD |     PORCENTAJE      |   PALABRAS MAS FRECUENTES                      |
|----------|-------------------------|----------|---------------------|------------------------------------------------|
|  **nc**  |  Nombre común           | 92002    |  17.78503769572782% | ['años', 'presidente', 'millones', 'equipo',   |
|          |                         |          |                     |  'partido']                                    |
|  **sp**  |  Adposición preposición | 79904    | 15.446356079644307% | ['de', 'en', 'a', 'del', 'con']                |
|  **da**  |  Determinante artículo  | 54552    | 10.545524840518075% | ['la', 'el', 'los', 'las', 'El']               |
|  **vm**  |  Verbo principal        | 50609    |  9.783297892905471% | ['está', 'tiene', 'dijo', 'puede', 'hace']     |
|  **aq**  |  Adjetivo calificativo  | 33904    |  6.554030543205104% | ['pasado', 'gran', 'mayor', 'nuevo', 'próximo']|
|  **fc**  |  Puntuación coma        | 30148    |  5.827952832012372% | [',']                                          |
|  **np**  |  Nombre propio          | 29113    |  5.627875507442489% | ['Gobierno', 'España', 'PP', 'Barcelona',      |
|          |                         |          |                     |  'Madrid']                                     |
|  **fp**  |  Puntuación (punto,     | 21157    |  4.089889812487918% | ['.', '(', ')']                                |
|          |  parentesis)            |          |                     |                                                |
|  **rg**  |  Adverbio general       | 15333    |  2.964044075004833% | ['más', 'hoy', 'también', 'ayer', 'ya']        |
|  **cc**  |  Conjunción coordinada  | 15023    |  2.904117533346221% | ['y', 'pero', 'o', 'Pero', 'e']                |


##Ejercicio 3: Entrenamiento y Evaluación de Taggers

100.0% (89.03%)
Precisión: 89.03%.
Precisión - palabras conocidas: 95.35%.
Precisión - palabras desconocidas:  31.80%.

real    0m4.580s.
user    0m4.441s.
sys 0m0.107s.


##Ejercicio 5: HMM POS Tagger

n = 1

100.0% (89.01%).
Precisión: 89.01%.
Precisión - palabras conocidas: 95.32%.
Precisión - palabras desconocidas:  31.80%.

real    0m21.850s.
user    0m21.099s.
sys 0m0.283s.


n = 2

100.0% (92.72%).
Precisión: 92.72%.
Precisión - palabras conocidas: 97.61%.
Precisión - palabras desconocidas:  48.42%.

real    0m27.572s.
user    0m27.147s.
sys 0m0.295s.

n = 3

100.0% (93.17%).
Precisión: 93.17%.
Precisión - palabras conocidas: 97.67%.
Precisión - palabras desconocidas:  52.31%.

real    1m28.704s.
user    1m27.217s.
sys 0m1.238s.

n = 4

100.0% (93.14%).
Precisión: 93.14%.
Precisión - palabras conocidas: 97.44%.
Precisión - palabras desconocidas:  54.16%.

real    10m55.385s.
user    10m51.024s.
sys 0m1.980s.


##Ejercicio 7

- LogisticRegression

n = 1

100.0% (92.70%).
Precisión: 92.70%.
Precisión - palabras conocidas: 95.28%.
Precisión - palabras desconocidas:  69.32%.

real    0m50.751s.
user    0m39.538s.
sys 0m0.576s.


n = 2

100.0% (91.99%).
Precisión: 91.99%.
Precisión - palabras conocidas: 94.55%.
Precisión - palabras desconocidas:  68.75%.

real    0m47.429s.
user    0m43.195s.
sys 0m0.392s.

n = 3

100.0% (92.18%).
Precisión: 92.18%.
Precisión - palabras conocidas: 94.72%.
Precisión - palabras desconocidas:  69.20%.

real    1m19.324s.
user    1m14.270s.
sys 0m0.719s.

n = 4

100.0% (92.22%).
Precisión: 92.22%.
Precisión - palabras conocidas: 94.72%.
Precisión - palabras desconocidas:  69.59%.

real    1m45.489s.
user    1m33.383s.
sys 0m0.902s.


- MultinomialNB

n = 1

100.0% (82.18%).
Precisión: 82.18%.
Precisión - palabras conocidas: 85.85%.
Precisión - palabras desconocidas:  48.89%.

real    44m37.125s.
user    43m52.380s.
sys 0m4.104s.

n = 2

100.0% (76.46%).
Precisión: 76.46%.
Precisión - palabras conocidas: 80.41%.
Precisión - palabras desconocidas:  40.68%.

real    44m16.972s.
user    43m44.351s.
sys 0m1.796s.

n = 3

100.0% (71.47%).
Precisión: 71.47%.
Precisión - palabras conocidas: 75.09%.
Precisión - palabras desconocidas:  38.59%.

real    44m39.067s.
user    44m8.322s.
sys 0m2.267s.

n = 4

100.0% (68.20%).
Precisión: 68.20%.
Precisión - palabras conocidas: 71.31%.
Precisión - palabras desconocidas:  40.01%.

real    21m40.869s.
user    21m37.278s.
sys 0m0.687s.


- LinearSVC

n = 1

100.0% (94.43%).
Precisión: 94.43%.
Precisión - palabras conocidas: 97.04%.
Precisión - palabras desconocidas:  70.82%.

real    0m44.183s.
user    0m43.581s.
sys 0m0.367s.

n = 2

100.0% (94.29%).
Precisión: 94.29%.
Precisión - palabras conocidas: 96.91%.
Precisión - palabras desconocidas:  70.57%.

real    0m35.505s.
user    0m35.161s.
sys 0m0.252s.

n = 3

100.0% (94.40%).
Precisión: 94.40%.
Precisión - palabras conocidas: 96.94%.
Precisión - palabras desconocidas:  71.38%.

real    0m43.238s.
user    0m42.628s.
sys 0m0.403s.

n = 4

100.0% (94.46%).
Precisión: 94.46%.
Precisión - palabras conocidas: 96.96%.
Precisión - palabras desconocidas:  71.81%.

real    0m43.377s.
user    0m37.706s.
sys 0m0.620s.
